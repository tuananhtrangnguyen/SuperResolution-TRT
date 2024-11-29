cmake_minimum_required(VERSION 3.10)
project(sr LANGUAGES CXX CUDA)

# Cấu hình đường dẫn CMake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Thiết lập cờ biên dịch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Thêm thư mục chứa header file của dự án
include_directories(${PROJECT_SOURCE_DIR}/src/include)

# Tìm và liên kết CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-10.2/include)  # Sử dụng đường dẫn chung cho CUDA
link_directories(/usr/local/cuda-10.2/lib64)

# Tìm và liên kết OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Liên kết TensorRT
# Nếu TensorRT không được tìm thấy tự động, bạn cần thêm đường dẫn thủ công
include_directories(/usr/include/x86_64-linux-gnu/NVIDIA/)  # Đường dẫn cho header TensorRT
link_directories(/usr/lib/aarch64-linux-gnu/)  # Đường dẫn cho thư viện TensorRT

# Thêm NVTX
include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda-10.2/include/nvtx3)

# Thêm các tệp nguồn
set(SOURCES
    main.cpp
)

# Tạo tệp thực thi và liên kết các thư viện cần thiết
add_executable(${PROJECT_NAME} ${SOURCES})

# Liên kết các thư viện với tệp thực thi
target_link_libraries(${PROJECT_NAME} PRIVATE 
    nvinfer 
    cudart
    nvToolsExt 
    ${OpenCV_LIBS} 
    dl)  # Thêm libdl vào đây
