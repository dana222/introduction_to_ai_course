from typing import Tuple, List, Optional
class GridWorld:
    def __init__(self, map_file: str):
        with open(map_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.height = len(lines)
        self.width = max(len(line) for line in lines)
        self.grid = np.zeros((self.height, self.width), dtype=int)

        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None

        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == "S":
                    self.start = (r, c)
                    self.grid[r, c] = 0
                elif char == "G":
                    self.goal = (r, c)
                    self.grid[r, c] = 0
                elif char == "1":
                    self.grid[r, c] = 1  # obstacle
                else:
                    self.grid[r, c] = 0  # free cell

        if self.start is None or self.goal is None:
            raise ValueError("Map file must contain one start 'S' and one goal 'G'.")

    def is_valid(self, position: Tuple[int, int]) -> bool:
        r, c = position
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0

    def display(self):
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                if (r, c) == self.start:
                    row_str += "S "
                elif (r, c) == self.goal:
                    row_str += "G "
                elif self.grid[r, c] == 1:
                    row_str += "1 "
                else:
                    row_str += "0 "
            print(row_str)

def demo():
    print("Demo function - not implemented")
