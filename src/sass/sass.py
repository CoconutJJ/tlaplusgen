from cleaner import clean
from parser import parse_file
from cfg import build_cfg
from argparse import ArgumentParser
from slicer import slice_sass

args = ArgumentParser()
if __name__ == "__main__":

    subparser = args.add_subparsers(dest="command")

    sass_cleaner = subparser.add_parser("sass-clean")
    sass_cleaner.add_argument("file")
    sass_cleaner.add_argument("--out")

    parse = subparser.add_parser("sass-parse")
    parse.add_argument("file")
    parse.add_argument("--out")

    params = args.parse_args()

    match params.command:
        case "sass-clean":
            with open(params.file, "r") as f:
                code = f.read()
            
            code = clean(code)

            if params.out is not None:
                with open(params.out, "w") as f:
                    f.write(code)
            else:
                print(code)
        case "sass-parse":
            with open(params.file, "r") as f:
                code = f.read()
            
            with open("out.dot", "w") as f:
                cfg = slice_sass(code, ["re:WARPSYNC"])
                cfg.dump_dot(f)



    


