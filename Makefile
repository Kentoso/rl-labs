.PHONY: demo-value demo-policy demo-mc demo-mc-unshaped

# Seeds to play, e.g. make demo-mc SEEDS="1234 5"
SEEDS ?= 1234 0 1 2

demo-value:
	uv run python lab1/demo.py --method value --seeds $(SEEDS)

demo-policy:
	uv run python lab1/demo.py --method policy --seeds $(SEEDS)

# Any saved MC policy, e.g. make demo-mc POLICY=vel0.6_pos0.2
POLICY ?= vel0.8_pos0
demo-mc:
	uv run python lab2/demo.py --policy $(POLICY) --seeds $(SEEDS)

demo-mc-unshaped:
	uv run python lab2/demo.py --policy vel0_pos0 --seeds $(SEEDS)
