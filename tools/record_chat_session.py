"""Drive chat.py inside a real pseudo-terminal and save the raw session log.

This is the actual terminal interface from the assignment starter, running as a
real interactive process: prompts are typed into a pty, the model answers, and
everything the terminal printed is captured verbatim to a .txt log alongside
chat.py's own JSON transcript. Nothing is simulated or re-typed afterwards.

    python tools/record_chat_session.py --model experiments/expanded/llm_run/model.pt \
        --transcript results/chat/expanded_chat_transcript.json \
        --log results/chat/expanded_terminal_session.txt
"""
import argparse
import sys
from pathlib import Path

import pexpect

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "the opposite of heavy is",
    "the pen is inside the jar . the jar contains the",
    "yesterday the baker",
    "the cup is not red . it is green . the cup is",
    "what do you think about the french revolution",
    "the team discussed the loan and the interest at the",
    ("the customer and the buyer and the shopper and the client and the consumer and the "
     "subscriber walked into the store and asked about the price and the delivery and the "
     "design and the quality of the package and the item and the brand and the offering and "
     "the merchandise before the order was finally ready to collect at the market"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    child = pexpect.spawn(sys.executable, ["chat.py", "--model", args.model,
                                           "--transcript", args.transcript],
                          cwd=str(ROOT), encoding="utf-8", timeout=120, echo=True,
                          dimensions=(40, 100))
    child.logfile_read = open(args.log, "w", encoding="utf-8")
    for prompt in PROMPTS:
        child.expect("You: ")
        child.sendline(prompt)
    child.expect("You: ")
    child.sendline("/quit")
    child.expect(pexpect.EOF)
    child.logfile_read.close()
    child.close()
    print("Session log:", args.log)
    print("Transcript :", args.transcript)
    return child.exitstatus or 0


if __name__ == "__main__":
    sys.exit(main())
