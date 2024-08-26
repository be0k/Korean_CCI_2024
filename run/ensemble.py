import pandas as pd
import json
from collections import Counter
import argparse
import os

parser = argparse.ArgumentParser(prog="ensemble", description="ensemble about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--dir",type=str, default=f'output',  help="output filename")
g.add_argument("--d0",type=str, default=f'fold0_quan.json',  help="output filename 0")
g.add_argument("--d1",type=str, default=f'fold1_quan.json',  help="output filename 1")
g.add_argument("--d2",type=str, default=f'fold2_quan.json',  help="output filename 2")
g.add_argument("--d3",type=str, default=f'fold3_quan.json',  help="output filename 3")
g.add_argument("--d4",type=str, default=f'fold4_quan.json',  help="output filename 4")
g.add_argument("--d5",type=str, default=f'fold5_quan.json',  help="output filename 5")
g.add_argument("--d6",type=str, default=f'fold6_quan.json',  help="output filename 6")
g.add_argument("--d7",type=str, default=f'fold7_quan.json',  help="output filename 7")
g.add_argument("--d8",type=str, default=f'fold8_quan.json',  help="output filename 8")
g.add_argument("--d9",type=str, default=f'fold9_quan.json',  help="output filename 9")
g.add_argument("--d10",type=str, default=f'orig_quan.json',  help="output filename 10")
g.add_argument("--save_dir",type=str, default=f'ensemble.json',  help="save file name")


def main(args):
    with open(os.path.join(args.dir, args.d0), "r", encoding="utf-8") as f:
        r0 = json.load(f)

    with open(os.path.join(args.dir, args.d1), "r", encoding="utf-8") as f:
        r1 = json.load(f)

    with open(os.path.join(args.dir, args.d2), "r", encoding="utf-8") as f:
        r2 = json.load(f)

    with open(os.path.join(args.dir, args.d3), "r", encoding="utf-8") as f:
        r3 = json.load(f)

    with open(os.path.join(args.dir, args.d4), "r", encoding="utf-8") as f:
        r4 = json.load(f)

    with open(os.path.join(args.dir, args.d5), "r", encoding="utf-8") as f:
        r5 = json.load(f)

    with open(os.path.join(args.dir, args.d6), "r", encoding="utf-8") as f:
        r6 = json.load(f)

    with open(os.path.join(args.dir, args.d7), "r", encoding="utf-8") as f:
        r7 = json.load(f)

    with open(os.path.join(args.dir, args.d8), "r", encoding="utf-8") as f:
        r8 = json.load(f)

    with open(os.path.join(args.dir, args.d9), "r", encoding="utf-8") as f:
        r9 = json.load(f)

    with open(os.path.join(args.dir, args.d10), "r", encoding="utf-8") as f:
        r10 = json.load(f)

    for i, (a,b,c,d,e,f,g,h,j,k,l) in enumerate(zip(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10)):
        variables = [a['output'], b['output'], c['output'], d['output'], e['output'], f['output'], g['output'], h['output'], j['output'], k['output'],l['output']]

        # 모든 변수가 동일한지 확인합니다.
        if all(x == variables[0] for x in variables):
            # 모든 변수가 동일하면 그 값을 유지합니다.

            r10[i]['output'] = variables[0]
        else:
            qqq = Counter(variables)
            least_common = max(qqq, key=qqq.get)
            r10[i]['output'] = least_common

    with open(os.path.join(args.dir, args.save_dir), "w", encoding="utf-8") as f:
        f.write(json.dumps(r10, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    exit(main(parser.parse_args()))