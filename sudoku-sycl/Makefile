FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wall -Wno-linker-warnings
BUILD_DIR = build

SOURCE_FILES = main.cpp

all: $(BUILD_DIR)/sudoku.exe

$(BUILD_DIR)/sudoku.exe: $(SOURCE_FILES)
	clang++ $(FLAGS) -o $@ $^

run:
	$(BUILD_DIR)/sudoku.exe

clean:
	del $(BUILD_DIR)

.PHONY: all clean run
