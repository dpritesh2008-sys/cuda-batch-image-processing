#include "image_processing.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace fs=std::filesystem;

static void Check(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
__device__ __forceinline__ unsigned char Clamp(float v){return (unsigned char)fminf(255.0f,fmaxf(0.0f,v));}

__global__ void SobelBatchKernel(const unsigned char* in,unsigned char* out,int w,int h,int n){
  long long total=(long long)w*h*n;
  for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=(long long)gridDim.x*blockDim.x){
    long long imgpx=(long long)w*h, local=i%imgpx, base=(i/imgpx)*imgpx;
    int y=local/w,x=local%w;
    if(x==0||y==0||x==w-1||y==h-1){out[i]=0;continue;}
    int a=in[base+(y-1)*w+x-1],b=in[base+(y-1)*w+x],c=in[base+(y-1)*w+x+1];
    int d=in[base+y*w+x-1],f=in[base+y*w+x+1];
    int g=in[base+(y+1)*w+x-1],h1=in[base+(y+1)*w+x],j=in[base+(y+1)*w+x+1];
    int gx=-a+c-2*d+2*f-g+j, gy=-a-2*b-c+g+2*h1+j;
    out[i]=Clamp(sqrtf((float)(gx*gx+gy*gy)));
  }
}
static std::string Tok(std::istream& s){std::string t;while(s>>t){if(t[0]=='#'){std::string z;std::getline(s,z);continue;}return t;}return{};}

Image ReadPgm(const std::string& p){
  std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot open "+p);
  if(Tok(f)!="P5")throw std::runtime_error("Expected P5 PGM");
  Image im;im.width=std::stoi(Tok(f));im.height=std::stoi(Tok(f));int maxv=std::stoi(Tok(f));
  if(im.width<=0||im.height<=0||maxv!=255)throw std::runtime_error("Unsupported PGM");
  f.get();im.pixels.resize((size_t)im.width*im.height);
  f.read((char*)im.pixels.data(),im.pixels.size());
  if(f.gcount()!=(std::streamsize)im.pixels.size())throw std::runtime_error("Incomplete PGM");
  return im;
}
void WritePgm(const std::string& p,const Image& im){
  std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot write "+p);
  f<<"P5\n"<<im.width<<" "<<im.height<<"\n255\n";f.write((char*)im.pixels.data(),im.pixels.size());
}
std::vector<std::string> FindPgmFiles(const std::string& dir,size_t limit){
  std::vector<std::string> v;if(!fs::exists(dir))throw std::runtime_error("Input directory not found");
  for(auto&e:fs::directory_iterator(dir))if(e.is_regular_file()&&e.path().extension()==".pgm")v.push_back(e.path().string());
  std::sort(v.begin(),v.end());if(limit&&v.size()>limit)v.resize(limit);return v;
}
void RunSobelCuda(const std::vector<unsigned char>& in,std::vector<unsigned char>*out,int w,int h,int n,int threads,float*ms){
  size_t bytes=in.size();unsigned char *di=nullptr,*do_=nullptr;cudaEvent_t a=nullptr,b=nullptr;
  Check(cudaMalloc(&di,bytes),"cudaMalloc input");Check(cudaMalloc(&do_,bytes),"cudaMalloc output");
  try{
    Check(cudaMemcpy(di,in.data(),bytes,cudaMemcpyHostToDevice),"copy input");
    Check(cudaEventCreate(&a),"event");Check(cudaEventCreate(&b),"event");
    long long total=(long long)w*h*n;int blocks=(int)std::min(65535LL,(total+threads-1)/threads);
    Check(cudaEventRecord(a),"start");SobelBatchKernel<<<blocks,threads>>>(di,do_,w,h,n);
    Check(cudaGetLastError(),"kernel launch");Check(cudaEventRecord(b),"stop");Check(cudaEventSynchronize(b),"sync");
    Check(cudaEventElapsedTime(ms,a,b),"elapsed");out->resize(in.size());
    Check(cudaMemcpy(out->data(),do_,bytes,cudaMemcpyDeviceToHost),"copy output");
  }catch(...){if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);cudaFree(di);cudaFree(do_);throw;}
  cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(di);cudaFree(do_);
}
int main(int argc,char**argv){
 try{
  std::string inDir="data/input",outDir="output";size_t limit=0;int threads=256;
  for(int i=1;i<argc;i++){std::string a=argv[i];auto val=[&](){if(i+1>=argc)throw std::runtime_error("Missing argument value");return std::string(argv[++i]);};
    if(a=="--input")inDir=val();else if(a=="--output")outDir=val();else if(a=="--limit")limit=std::stoull(val());else if(a=="--threads")threads=std::stoi(val());else throw std::runtime_error("Unknown argument: "+a);}
  if(threads<1||threads>1024)throw std::runtime_error("--threads must be 1..1024");
  auto paths=FindPgmFiles(inDir,limit);if(paths.empty())throw std::runtime_error("No PGM files found");
  Image first=ReadPgm(paths[0]);int w=first.width,h=first.height;std::vector<unsigned char> in;
  in.reserve((size_t)paths.size()*w*h);
  for(auto&p:paths){Image im=ReadPgm(p);if(im.width!=w||im.height!=h)throw std::runtime_error("Mismatched dimensions");in.insert(in.end(),im.pixels.begin(),im.pixels.end());}
  std::vector<unsigned char> out;float ms=0;RunSobelCuda(in,&out,w,h,(int)paths.size(),threads,&ms);fs::create_directories(outDir);
  size_t bytes=(size_t)w*h;for(size_t i=0;i<paths.size();i++){Image im;im.width=w;im.height=h;im.pixels.assign(out.begin()+i*bytes,out.begin()+(i+1)*bytes);WritePgm((fs::path(outDir)/(std::string("edge_")+std::to_string(i)+".pgm")).string(),im);}
  cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0),"device properties");
  std::cout<<"GPU image processing completed successfully\nImages processed: "<<paths.size()<<"\nDimensions: "<<w<<"x"<<h<<"\nTotal pixels: "<<in.size()<<"\nGPU: "<<prop.name<<"\nThreads per block: "<<threads<<"\nGPU kernel execution time: "<<ms<<" ms\nOutput directory: "<<outDir<<"\n";
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}return 0;
}