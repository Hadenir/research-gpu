#include "board.hpp"

Board::Board() {}

Board::Board(uint8_t* cells)
{
    for(int i = 0; i < CELLS_COUNT; i++)
        this->cells[i] = cells[i];
}

bool Board::check() const
{
    bool occurences[BOARD_SIZE] {};

    // Check rows.
    for(int r = 0; r < BOARD_SIZE; r++)
    {
        memset(occurences, false, BOARD_SIZE);
        for(int c = 0; c < BOARD_SIZE; c++)
        {
            uint8_t digit = cells[c + BOARD_SIZE * r];
            if(digit == 0) continue;
            if(occurences[digit - 1]) return false;

            occurences[digit - 1] = true;
        }
    }

    // Check columns.
    for(int c = 0; c < BOARD_SIZE; c++)
    {
        memset(occurences, false, BOARD_SIZE);
        for(int r = 0; r < BOARD_SIZE; r++)
        {
            uint8_t digit = cells[c + BOARD_SIZE * r];
            if(digit == 0) continue;
            if(occurences[digit - 1]) return false;

            occurences[digit - 1] = true;
        }
    }

    // Check boxes.
    for(int b = 0; b < BOARD_SIZE; b++)
    {
        memset(occurences, false, BOARD_SIZE);
        for(int i = 0; i < BOARD_SIZE; i++)
        {
            int c = (b % BOX_SIZE) * BOX_SIZE + i % BOX_SIZE;
            int r = (b / BOX_SIZE) * BOX_SIZE + i / BOX_SIZE;
            uint8_t digit = cells[c + BOARD_SIZE * r];
            if(digit == 0) continue;
            if(occurences[digit - 1]) return false;

            occurences[digit - 1] = true;
        }
    }

    // All checks passed.
    return true;
}

bool Board::check_cell(int index) const
{
    int c = index % BOARD_SIZE;
    int r = index / BOARD_SIZE;

    bool occurences[BOARD_SIZE] {};

    // Check row.
    for(int i = 0; i < BOARD_SIZE; i++)
    {
        uint8_t digit = cells[i + BOARD_SIZE * r];
        if(digit == 0) continue;
        if(occurences[digit - 1]) return false;

        occurences[digit - 1] = true;
    }

    memset(occurences, false, BOARD_SIZE);
    // Check column.
    for(int i = 0; i < BOARD_SIZE; i++)
    {
        uint8_t digit = cells[c + BOARD_SIZE * i];
        if(digit == 0) continue;
        if(occurences[digit - 1]) return false;

        occurences[digit - 1] = true;
    }

    memset(occurences, false, BOARD_SIZE);
    // Check box.
    for(int i = 0; i < BOARD_SIZE; i++)
    {
        int bc = c / BOX_SIZE;
        int br = r / BOX_SIZE;
        int b = bc + br * BOX_SIZE;
        int x = (b % BOX_SIZE) * BOX_SIZE + i % BOX_SIZE;
        int y = (b / BOX_SIZE) * BOX_SIZE + i / BOX_SIZE;

        uint8_t digit = cells[x + BOARD_SIZE * y];
        if(digit == 0) continue;
        if(occurences[digit - 1]) return false;

        occurences[digit - 1] = true;
    }

    // All checks passed.
    return true;
}

size_t Board::find_empty() const
{
    size_t index = 0;
    while(cells[index] != 0 && index < CELLS_COUNT) index++;
    return index;
}

std::ostream& operator<<(std::ostream& os, const Board& board)
{
    for(int r = 0; r < Board::BOARD_SIZE; r++)
    {
        for(int c = 0; c < Board::BOARD_SIZE; c++)
        {
            os << (+board.cells[c + Board::BOARD_SIZE * r]) << ' ';
        }
        os << '\n';
    }

    return os;
}

const sycl::stream& operator<<(const sycl::stream& os, const Board& board)
{
    for(int r = 0; r < Board::BOARD_SIZE; r++)
    {
        for(int c = 0; c < Board::BOARD_SIZE; c++)
        {
            os << (+board.cells[c + Board::BOARD_SIZE * r]) << ' ';
        }
        os << '\n';
    }

    return os;
}
