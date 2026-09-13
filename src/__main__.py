import argparse

def main() -> None:
    """Run the main programm and parse the command
    line arguments with default values."""
    parser = argparse.ArgumentParser(
        description="Take command-line argument and parse them by their flags."
        )

    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="set the path to the function definition (by "
                        "default: 'data/input/functions_definition.json')")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="set the path to the input definition (by "
                        "default: 'data/input/function_calling_tests.json')")
    parser.add_argument("--output",
                        default="data/output/function_calls.json",
                        help="set the path to the output definition (by "
                        "default: 'data/output/function_calls.json')")
    args = parser.parse_args()

if __name__ == "__main__":
    main()