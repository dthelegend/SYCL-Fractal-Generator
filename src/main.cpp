#include <iostream>
#include <opencv2/opencv.hpp>
#include <sycl/sycl.hpp>

// This number of Tiles produces a HD image
#define NUM_TILES_X 10
#define NUM_TILES_Y 10

// From my understanding, square tiles are the most efficient
#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define MAX_SIMULTANEOUS_TILES 1

// Argand Diagram Size
#define ARGAND_START_X -2.0f
#define ARGAND_END_X 1.0f
#define ARGAND_START_Y -1.0f
#define ARGAND_END_Y 1.0f

#define INTER_TILE_ARGAND_X (ARGAND_END_X - ARGAND_START_X) / TILE_SIZE_X
#define INTER_TILE_ARGAND_Y (ARGAND_END_Y - ARGAND_START_Y) / TILE_SIZE_Y

// Misc Options
#define MAX_ITERATIONS 100

int main(int argc, char const *argv[])
{
    // Create a temporary file to store the image
    std::FILE* tileCaches[NUM_TILES_X * NUM_TILES_Y];

    int num_tiles_left = NUM_TILES_X * NUM_TILES_Y;
    int current_tile = num_tiles_left;

    std::cout << "Total Number of tiles: " << num_tiles_left << std::endl;

    while (num_tiles_left > 0) {
        int number_of_tiles = std::min(num_tiles_left , MAX_SIMULTANEOUS_TILES);
        num_tiles_left -= number_of_tiles;

        std::cout << "Creating " << number_of_tiles << " tile buffers" << std::endl;

        // Create buffers to store the image
        sycl::float4 buffers[number_of_tiles][TILE_SIZE_X * TILE_SIZE_Y];
        
        sycl::queue Q(sycl::default_selector_v);
        
        std::cout << "Running on: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

        std::cout << "Queuing the rendering of " << number_of_tiles << " tiles" << std::endl;
        
        // Run the kernel
        for(int i = 0; i < number_of_tiles; i++) {
            int n_tile = --current_tile;
            int tile_x = n_tile % NUM_TILES_X;
            int tile_y = n_tile / NUM_TILES_X;

            // Create an image buffer to wrap our host buffer
            sycl::image<2> bufferImage(buffers[i], sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, sycl::range<2>(TILE_SIZE_X, TILE_SIZE_Y));
            
            std::cout << "Submitting tile " << n_tile << " at " << tile_x << ", " << tile_y << std::endl;

            Q.submit([&](sycl::handler& h) {
                // Create an accessor for the image
                auto outPtr = bufferImage.get_access<sycl::float4, sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<2>(TILE_SIZE_X, TILE_SIZE_Y), [=](sycl::item<2> item) {
                    sycl::int2 coords(item[0], item[1]);

                    std::complex<float> zn(0, 0);
                    std::complex<float> c(INTER_TILE_ARGAND_X * ((float) tile_x + ((float) coords[0]) / TILE_SIZE_X), INTER_TILE_ARGAND_Y * ((float) tile_y + ((float) coords[1]) / TILE_SIZE_Y));

                    int depth;
                    for (depth = 0; depth < MAX_ITERATIONS && abs(zn) <= 2; ++depth) {
                        zn = zn * zn;
                        zn = zn + c;
                    }
                    
                    // TODO: Colouring
                    // Black and White for now
                    if(depth < MAX_ITERATIONS) {
                        outPtr.write(coords, sycl::float4(0.0f, 0.0f, 0.0f, 1.0f));
                    } else {
                        outPtr.write(coords, sycl::float4(1.0f, 1.0f, 1.0f, 1.0f));
                    }
                });
            });
        }

        std::cout << "Rendering " << number_of_tiles << " tiles" << std::endl;

        Q.wait_and_throw();

        std::cout << "Finished Rendering " << number_of_tiles << " tiles" << std::endl << "Writing to file buffers" << std::endl;

        // Write the image to the file
        for(int i = 0; i < number_of_tiles; i++) {
            int n_tile = current_tile + i;
            int tile_x = n_tile % TILE_SIZE_X;
            int tile_y = n_tile / TILE_SIZE_X;

            tileCaches[n_tile] = std::tmpfile();
            cv::Mat image(NUM_TILES_Y * TILE_SIZE_Y, NUM_TILES_X * TILE_SIZE_X, CV_32FC4, buffers[i]);
            std::vector<uchar> buf;
            cv::imencode(".bmp", image, buf, std::vector<int>() );
            for (int i = 0; i < buf.size(); i++) {
                std::fputc(buf[i], tileCaches[n_tile]);
            }
        }

        std::cout << "Finished writing to file buffers" << std::endl;
    }

    std::cout << "Finished rendering all tiles" << std::endl << "Stiching the image together" << std::endl;

    // Use OpenCV to stich the images together
    cv::Mat finalImage(NUM_TILES_Y * TILE_SIZE_Y, NUM_TILES_X * TILE_SIZE_X, CV_32FC4);
    for(int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; i++) {
        std::rewind(tileCaches[i]);
        cv::Mat tile = cv::imdecode(cv::Mat(cv::Mat(1, std::ftell(tileCaches[i]), CV_8UC1, tileCaches[i])), cv::IMREAD_UNCHANGED);
        int tile_x = i % TILE_SIZE_X;
        int tile_y = i / TILE_SIZE_X;
        tile.copyTo(finalImage(cv::Rect(tile_x * NUM_TILES_X, tile_y * NUM_TILES_Y, NUM_TILES_X, NUM_TILES_Y)));
    }

    std::cout << "Finished stiching the image together" << std::endl << "Saving the image" << std::endl;

    // Save the image
    cv::imwrite("output.bmp", finalImage);

    std::cout << "Job Done!" << std::endl;
    return 0;
}
