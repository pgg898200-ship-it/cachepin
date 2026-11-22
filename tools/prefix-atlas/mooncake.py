"""Convert Mooncake JSONL to a weighted tree with the Prefix Atlas importer."""

from prefix_atlas.mooncake import Edge, PrefixTree, convert, main

__all__ = ["Edge", "PrefixTree", "convert", "main"]

if __name__ == "__main__":
    main()
