#pragma once

#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

struct Board
{
    const static size_t BOX_SIZE = 3;
    const static size_t BOARD_SIZE = BOX_SIZE * BOX_SIZE;
    const static size_t CELLS_COUNT = BOARD_SIZE * BOARD_SIZE;

    uint8_t cells[CELLS_COUNT] {};

    Board();
    explicit Board(uint8_t* cells);

    // Checks whole board for correctnes agaist Sudoku rules.
    SYCL_EXTERNAL bool check() const;

    // Checks is specified (filled) cell is valid.
    SYCL_EXTERNAL bool check_cell(int index) const;

    // Finds index to first not-filled cell in the board.
    // If board is filled completely, returns CELLS_COUNT.
    SYCL_EXTERNAL size_t find_empty() const;

    friend std::ostream& operator<<(std::ostream& os, const Board& board);
    friend const sycl::stream& operator<<(const sycl::stream& os, const Board& board);
};

std::ostream& operator<<(std::ostream& os, const Board& board);
SYCL_EXTERNAL const sycl::stream& operator<<(const sycl::stream& os, const Board& board);
