"""
Command-line entry for autodiff.

The ada command-line tool reads a file containing APL dfns,
computes their derivatives, and writes the results to a new file.

Usage:
    ada <apl> <aplparse>

Args:
    apl: Path to APL file of dfns to differentiate.
    aplparse: Path to aplparse executable.
"""


def main():
    import argparse
    import re

    from .autodiff import autodiff

    parser = argparse.ArgumentParser(description='Differentiates a file of APL dfns and \
                                                  writes the derivatives to a new file.')
    parser.add_argument('apl',
                        help='Path to APL file of dfns to differentiate.',
                        type=str)
    parser.add_argument('aplparse',
                        help='Path to aplparse executable.',
                        type=str)
    args = parser.parse_args()

    res = []
    with open(args.apl, 'r') as f:
        # The regex pattern extracts dfns.
        for apl in re.findall(r'\b\w+←\{[^{}]*\}', f.read()):
            print(f'Differentiating {apl.split("←")[0]}...')
            res.append(autodiff(apl, args.aplparse))
            print(f'Derivative successfully calculated.')

    # A 'd' prefix is prepended to the original filename.
    with open(re.sub(r'([^/]+)$', r'd\1', args.apl), 'w+') as f:
        f.write('\n\n'.join(res))


if __name__ == '__main__':
    main()
