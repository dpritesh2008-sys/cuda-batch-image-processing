#ifndef CUDA_BATCH_IMAGE_PROCESSING_IMAGE_PROCESSING_H_
#define CUDA_BATCH_IMAGE_PROCESSING_IMAGE_PROCESSING_H_

#include <cstddef>
#include <string>
#include <vector>

struct Image {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;
};

Image ReadPgm(const std::string& path);
void WritePgm(const std::string& path, const Image& image);
std::vector<std::string> FindPgmFiles(const std::string& directory,
                                      std::size_t limit);
void RunSobelCuda(const std::vector<unsigned char>& input,
                  std::vector<unsigned char>* output, int width, int height,
                  int image_count, int threads_per_block, float* elapsed_ms);

#endif  // CUDA_BATCH_IMAGE_PROCESSING_IMAGE_PROCESSING_H_
