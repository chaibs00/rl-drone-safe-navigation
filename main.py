"""SafeRoute entry point.

    python main.py train       Train the 4 agents over all seeds
    python main.py evaluate    Compute the metrics table      (Step 10)
    python main.py plot        Generate the figures           (Step 11)

Add --config to use another config file:
    python main.py train --config configs/default.yaml

    # Run test before : 
        python -m pytest tests/test_train.py -v
        python -m pytest tests/test_metrics.py -v

    # Then run main 
        python main.py train
        python main.py evaluate
"""


import argparse

from scripts import evaluate, train


def main():
    parser = argparse.ArgumentParser(description="SafeRoute: safe RL for drone delivery")
    parser.add_argument("command", choices=["train", "evaluate", "plot"],
                        help="what to run")
    parser.add_argument("--config", default="config.yaml",
                        help="path to the config file")
    args = parser.parse_args()

    if args.command == "train":
        train.main(args.config)
    elif args.command == "evaluate":
        evaluate.main(args.config)
    else:
        print(f"'{args.command}' is not implemented yet.")


if __name__ == "__main__":
    main()