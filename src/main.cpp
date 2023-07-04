#define SYCL_EXT_ONEAPI_COMPLEX

#include "main.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>                                           
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#include <sycl/sycl.hpp>

// This number of Tiles produces a HD image
#define NUM_TILES_X 512
#define NUM_TILES_Y 512

// From my understanding, square tiles are the most efficient
#define TILE_SIZE_X 64
#define TILE_SIZE_Y TILE_SIZE_X
#define MAX_SIMULTANEOUS_TILES 8

// Argand Diagram Size
#define ARGAND_START_X -2.0f
#define ARGAND_END_X 1.0f
#define ARGAND_START_Y -1.0f
#define ARGAND_END_Y 1.0f

// Misc Options
#define MAX_ITERATIONS 100

int main(int argc, char const *argv[])
{
    sycl::queue Q(sycl::default_selector_v);
    
    std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
    
    // Create a cv mat to store the image
    cv::Mat image(NUM_TILES_Y * TILE_SIZE_Y, NUM_TILES_X * TILE_SIZE_X, CV_32FC4);

    const int BATCH_SIZE = MAX_SIMULTANEOUS_TILES == -1 ? NUM_TILES_X * NUM_TILES_Y : MAX_SIMULTANEOUS_TILES;

    std::cout << "Rendering in total " << NUM_TILES_X * NUM_TILES_Y << " tiles at batch size " << BATCH_SIZE << std::endl;

    int n_tiles;
    int current_batch_size;
    for(n_tiles = NUM_TILES_X * NUM_TILES_Y; (current_batch_size = std::min(BATCH_SIZE, n_tiles)) > 0; n_tiles -= current_batch_size) {
        std::cout << "Creating " << current_batch_size << " tile buffers" << std::endl;

        // Create buffers to store the image
        sycl::float4 buffers[current_batch_size][TILE_SIZE_X * TILE_SIZE_Y];

        std::cout << "Queuing the rendering of " << current_batch_size << " tiles" << std::endl;
        
        // Run the kernel
        for(int i = 0; i < current_batch_size; i++) {
            int n_tile = n_tiles - i - 1;
            int tile_x = n_tile % NUM_TILES_X;
            int tile_y = n_tile / NUM_TILES_X;

            // Create an image buffer to wrap our host buffer
            sycl::image<2> bufferImage(buffers[i], sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, sycl::range<2>(TILE_SIZE_X, TILE_SIZE_Y));
            
            std::cout << "Submitting tile " << n_tile << " at " << tile_x << ", " << tile_y << std::endl;

            Q.submit([&](sycl::handler &h) {
                // Create an accessor for the image
                sycl::accessor<sycl::float4, 2, sycl::access::mode::write, sycl::access::target::image> outPtr(bufferImage, h);

                h.parallel_for(sycl::range<2>(TILE_SIZE_Y, TILE_SIZE_X), [=](sycl::item<2> item) {
                    sycl::int2 coords(item[0], item[1]);

                    sycl::ext::oneapi::experimental::complex<float> zn(0, 0);
                    sycl::ext::oneapi::experimental::complex<float> c(ARGAND_START_X + (((ARGAND_END_X - ARGAND_START_X) * (tile_x * TILE_SIZE_X + coords[0])) / (TILE_SIZE_X * NUM_TILES_X)), ARGAND_START_Y + (((ARGAND_END_Y - ARGAND_START_Y) * (tile_y * TILE_SIZE_Y + coords[1])) / (TILE_SIZE_Y * NUM_TILES_Y)));

                    int depth;
                    for (depth = 0; depth < MAX_ITERATIONS && abs(zn) <= 2; ++depth) {
                        zn = zn * zn;
                        zn = zn + c;
                    }
                    
                    // TODO: Colouring
                    // Black and White for now
                    // if(depth < MAX_ITERATIONS) {
                    //     outPtr.write(coords, sycl::float4(0.0f, 0.0f, 0.0f, 1.0f));
                    // } else {
                    //     outPtr.write(coords, sycl::float4(1.0f, 1.0f, 1.0f, 1.0f));
                    // }
                    float color = (float) depth / (float) MAX_ITERATIONS;
                    outPtr.write(coords, sycl::float4(color, color, color, 1.0f));
                });
            });
        }


        std::cout << "Rendering " << current_batch_size << " tiles..." << std::endl;

        Q.wait_and_throw();

        std::cout << "Finished Rendering " << current_batch_size << " tiles" << std::endl << "Writing to OpenCV Mat buffer" << std::endl;

        // Write the image to the mat
        for(int i = 0; i < current_batch_size; i++) {
            int n_tile = n_tiles - i - 1;
            int tile_x = n_tile % NUM_TILES_X;
            int tile_y = n_tile / NUM_TILES_X;

            std::cout << "Generating subtile " << n_tile << " at " << tile_x << ", " << tile_y << "..." << std::endl;

            cv::Mat subImage(TILE_SIZE_Y, TILE_SIZE_X, CV_32FC4, buffers[i]);
            subImage.convertTo(subImage, CV_8UC4, 255.0f);
            // cv::imshow("Sub Image", subImage);
            // cv::waitKey(0);
            // cv::destroyAllWindows();

            std::cout << "Writing subtile " << n_tile << " at " << tile_x * TILE_SIZE_X << ", " << tile_y * TILE_SIZE_Y << "..." << std::endl;

            subImage.copyTo(image(cv::Rect(tile_x * TILE_SIZE_X, tile_y * TILE_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y)));
        }

        std::cout << "Finished writing to OpenCV Mat buffer" << std::endl;
    }

    std::cout << "Finished rendering all tiles" << std::endl << "Saving the image" << std::endl;

    // Save the image
    cv::imwrite("output.bmp", image);

    std::cout << "Job Done!" << std::endl;
    return 0;
}
