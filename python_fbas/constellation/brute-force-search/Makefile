# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g -MMD -MP -O3
LDFLAGS =

# Source files for each program
SRC1 = src/partitions.c
SRC2 = src/multi-partitions.c src/optimal_cluster_assignment.c

# Generate object and dependency files
OBJ1 = $(SRC1:.c=.o)
OBJ2 = $(SRC2:.c=.o)
DEP1 = $(OBJ1:.o=.d)
DEP2 = $(OBJ2:.o=.d)

# Target binaries
TARGET1 = partitions
TARGET2 = optimal_cluster_assignment

# Default target builds both programs
all: $(TARGET1) $(TARGET2)

install: $(TARGET2)
	if [ -n "$(PREFIX)" ]; then \
		mkdir -p $(PREFIX); \
		cp $(TARGET2) $(PREFIX)/; \
		echo "Executable installed to $(PREFIX)/"; \
	elif [ -n "$(VIRTUAL_ENV)" ]; then \
		cp $(TARGET2) $(VIRTUAL_ENV)/bin/; \
		echo "Executable installed to $(VIRTUAL_ENV)/bin/"; \
	else \
		echo "No installation path provided and no virtual environment detected! Aborting."; \
		exit 1; \
	fi

# Rule to build program1
$(TARGET1): $(OBJ1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build program2
$(TARGET2): $(OBJ2)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Include dependency files
-include $(DEP1) $(DEP2)

# Clean up generated files
clean:
	rm -f $(OBJ1) $(OBJ2) $(DEP1) $(DEP2) $(TARGET1) $(TARGET2)

.PHONY: all clean install

