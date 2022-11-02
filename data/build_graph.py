import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generates graphs given number of nodes and density"
    )
    parser.add_argument("-n", "--nodes", type=int, default=1000)
    parser.add_argument("-d", "--density", type=float, default=0.05)

    args = parser.parse_args()
