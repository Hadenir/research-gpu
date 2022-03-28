FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wall -Wno-linker-warnings
BUILD_DIR = build

SOURCE_FILES = main.cpp board.cpp

all: $(BUILD_DIR)/sudoku.exe

$(BUILD_DIR)/%.o: %.cpp
	clang++ $(FLAGS) -c -o $@ $^

$(BUILD_DIR)/sudoku.exe: $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SOURCE_FILES))
	clang++ $(FLAGS) -o $@ $^

run:
	$(BUILD_DIR)/sudoku.exe boards/testboard.txt

clean:
	del $(BUILD_DIR)

.PHONY: all clean run
