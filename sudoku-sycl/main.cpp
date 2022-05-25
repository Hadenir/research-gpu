#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <fstream>
#include <chrono>
#include <sycl/sycl.hpp>

#include "board.hpp"
#include "cuda_selector.hpp"

namespace sycl = cl::sycl;

std::vector<uint8_t> read_file(const std::string& file_path)
{
    std::ifstream file(file_path);
    if(!file.is_open())
        throw std::runtime_error("Failed to open file " + file_path);

    std::vector<uint8_t> cells(Board::CELLS_COUNT, 0);

    int i = 0;
    while(!file.eof() && i < Board::CELLS_COUNT)
    {
        int digit;
        file >> digit;
        cells[i++] = digit;
    }

    if(i != Board::CELLS_COUNT)
        throw std::runtime_error("File doesn't contain the whole board");

    return cells;
}

void run_sudoku_bfs(
    sycl::queue& queue,
    size_t num_input_boards,
    sycl::buffer<Board>& input_boards_buf,
    sycl::buffer<size_t>& num_output_boards_buf,
    sycl::buffer<Board>& output_boards_buf,
    sycl::buffer<uint8_t>& empty_indices_buf,
    sycl::buffer<uint8_t>& num_empty_indices_buf)
{
    num_output_boards_buf.get_access<sycl::access_mode::discard_write>()[0] = 0;

    queue.submit([&](sycl::handler& h) {
        auto input_boards = input_boards_buf.get_access<sycl::access_mode::read_write>(h);
        auto output_boards = output_boards_buf.get_access<sycl::access_mode::write>(h);
        auto num_output_boards = num_output_boards_buf.get_access<sycl::access_mode::atomic>(h);
        auto empty_indices = empty_indices_buf.get_access<sycl::access_mode::write>(h);
        auto num_empty_indices = num_empty_indices_buf.get_access<sycl::access_mode::write>(h);

        h.parallel_for<class sudoku_bfs>(num_input_boards, [=](size_t idx) {
            Board& board = input_boards[idx];
            size_t cell_idx = board.find_empty();
            if(cell_idx >= Board::CELLS_COUNT) return;

            for(uint8_t digit = 1; digit <= Board::BOARD_SIZE; digit++)
            {
                board.cells[cell_idx] = digit;
                if(board.check_cell(cell_idx))
                {
                    size_t output_idx = num_output_boards[0].fetch_add(1);
                    uint8_t num_empty = 0;
                    for(size_t i = 0; i < Board::CELLS_COUNT; i++)
                    {
                        if(board.cells[i] == 0)
                            empty_indices[num_empty++ + Board::CELLS_COUNT * output_idx] = i;
                    }

                    num_empty_indices[output_idx] = num_empty;
                    output_boards[output_idx] = board;
                }
            }
        });
    });
}

void run_sudoku_backtrack(
    sycl::queue& queue,
    size_t num_boards,
    sycl::buffer<Board>& boards_buf,
    sycl::buffer<uint8_t>& empty_indices_buf,
    sycl::buffer<uint8_t>& num_empty_indices_buf,
    sycl::buffer<int>& complete_flag_buf,
    sycl::buffer<Board>& result_buf)
{
    complete_flag_buf.get_access<sycl::access_mode::write>()[0] = 0;

    queue.submit([&](sycl::handler& h) {
        auto boards = boards_buf.get_access<sycl::access_mode::read_write>(h);
        auto empty_indices = empty_indices_buf.get_access<sycl::access_mode::read>(h);
        auto num_empty_indices = num_empty_indices_buf.get_access<sycl::access_mode::read>(h);
        auto complete_flag = complete_flag_buf.get_access<sycl::access_mode::atomic>(h);
        auto result = result_buf.get_access<sycl::access_mode::write>(h);

        h.parallel_for<class sudoku_backtrack>(num_boards, [=] (size_t idx) {
            Board current_board = boards[idx];
            const uint8_t* current_empty_indices = empty_indices.get_pointer() + idx * Board::CELLS_COUNT;
            const uint8_t current_num_empty = num_empty_indices[idx];

            auto i = 0;
            while(i >= 0 && i < current_num_empty)
            {
                uint8_t current_empty = current_empty_indices[i];

                if(current_board.cells[current_empty] >= 9)
                {
                    current_board.cells[current_empty] = 0;
                    i--;
                    continue;
                }

                current_board.cells[current_empty] += 1;
                if(current_board.check_cell(current_empty)) i++;
            }

            if(i == current_num_empty && !complete_flag[0].exchange(1))
            {
                result[0] = current_board;
            }
        });
    });
}

int main(int argc, char* argv[])
{
    if(argc != 2) return -1;

    char* file_path = argv[1];
    auto cells = read_file(file_path);

    Board board(cells.data());

    const int num_bfs_steps = 21;
    const size_t max_num_boards = 2 << num_bfs_steps;

    // CudaSelector device_selector;
    sycl::host_selector device_selector;
    sycl::queue queue(device_selector);
    std::cout << "Running on device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::buffer<Board> input_boards_buf(max_num_boards);
    sycl::buffer<Board> output_boards_buf(max_num_boards);
    sycl::buffer<size_t> num_output_boards_buf(1);
    sycl::buffer<uint8_t> empty_indices_buf(Board::CELLS_COUNT * max_num_boards);
    sycl::buffer<uint8_t> num_empty_indices_buf(max_num_boards);

    input_boards_buf.get_access<sycl::access_mode::discard_write>()[0] = board;

    size_t num_boards = 1;
    for(auto i = 0; i < num_bfs_steps; i++)
    {
        if(i % 2 == 0)
            run_sudoku_bfs(queue, num_boards, input_boards_buf, num_output_boards_buf, output_boards_buf, empty_indices_buf, num_empty_indices_buf);
        else
            run_sudoku_bfs(queue, num_boards, output_boards_buf, num_output_boards_buf, input_boards_buf, empty_indices_buf, num_empty_indices_buf);

        num_boards = num_output_boards_buf.get_access<sycl::access_mode::read>()[0];
    }

    sycl::buffer<Board>& boards_buf = num_bfs_steps % 2 == 0
                                        ? input_boards_buf
                                        : output_boards_buf;
    sycl::buffer<int> complete_flag_buf(1);
    sycl::buffer<Board> result_buf(1);

    auto start = std::chrono::high_resolution_clock::now();
    run_sudoku_backtrack(queue, num_boards, boards_buf, empty_indices_buf, num_empty_indices_buf, complete_flag_buf, result_buf);
    queue.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();

    float duration_ms = std::chrono::duration(end - start).count() / 1000000.0f;

    auto complete = complete_flag_buf.get_access<sycl::access_mode::read>();
    auto result = result_buf.get_access<sycl::access_mode::read>();

    std::cout << "Time elapsed: " << duration_ms << "ms" << std::endl;
    std::cout << "Result: " << complete[0] << std::endl;
    std::cout << result[0] << std::endl;

    return 0;
}
