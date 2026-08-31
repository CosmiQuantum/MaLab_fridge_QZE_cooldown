"""Print compact summaries from Combined_Optimization_Scriptsv2 HDF5 files."""
import glob
import os
import sys

import h5py


def main():
    folder = sys.argv[1]
    for path in sorted(glob.glob(os.path.join(folder, "*.h5"))):
        with h5py.File(path, "r") as handle:
            attrs = {key: value.item() if hasattr(value, "item") else value
                     for key, value in handle.attrs.items()}
        print(os.path.basename(path), attrs)


if __name__ == "__main__":
    main()
