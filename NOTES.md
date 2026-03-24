Weak fairness: Something must be true infinitely often
- [F,T,F,T,F,T,...] is weakly fair. (For any T at index j in this list, there is
  another T at j' > j)
- In TLA+: []<>(E) for some expression E

Strong fairness: Something must be eventually permanently true
- [F,F,T,F,F,F,F,T,T,T,T,T,T,...,T,...] is strongly fair. (The index of the last 
  F in this list is a finite value)
- In TLA+: <>[](E) for some expression E

I've been playing around with the Always ([]), Eventually (<>), and Leads To
(~>) temporal operators in TLA+. My goal here is encode the idea that every
thread must eventually execute some instruction X.

So if we have the program counter pc_i for each thread i, then I think the
property in TLA+ should be written as

1. (<>(pc_1 = X) /\ <>(pc_2 = X) /\ <>(pc_3 = X) /\ <>(pc_4 = X) /\ ...)

This slightly abuses notation as X is an instruction, but pc should be a state.
But you get the idea.

In our case, we also need to express that each thread will eventually execute
some instruction, Y, after X as well. In other words, pc = X eventually leads to
pc = Y. For instance, after a thread calls USETMAXNREG, it must then call
WARPSYNC. The order that X is first executed before Y matters here. We also must
guarantee that there isn't another USETMAXNREG between X and all USETMAXNREG Z. 
In other words, for all USETMAXNREG instructions Z it is not eventually true 
that X leads to Z and it is eventually true that Z leads to Y where Z.

So the TLA+ representation should be:

2. /\ <>((Pc = X) ~>(pc = Y)) 
/\ (\A Z \in REG_INST : ~<>((Pc = X)~>(Pc = Z)) /\ <>((Pc =Z) ~> (Pc=Y)))

I've also been looking at this idea of weak fairness as it pertains to the
project. In this context, it is the idea that each thread must make progress
when progress is available.

We must first define what counts as "progress". I think it is reasonable to
define this as when some register value or the program counter has changed (when
we include memory refs as well, this definition may need to be amended).

So at any given state, we "make progress" when (regs' /= reg) \/ (pc' /= pc).
Furthermore, "progress is available" when the Next state expression is true. 

Thus, Next => (regs' /= regs) \/ (pc' /= pc). But it can't just be true at some
state, it must be true at every state. 

So it is actually: It is always the case that we make progress whenever progress
is available: []<>(Next /\ ((regs' /= reg) \/ (pc' /= pc)))

The above logic may prevent deadlock between two USETMAXNREG instructions, but
to be sure we introduce the following condition. For all X, Y, and Z that are
USETMAXNREG instructions, X eventually leads to Y and it is not thte case the 
X leads to Z which leads to Y, and Y leads to X


3. \A X \in REG_INST \A Y \in REG_INST \A Z \in REG_INST <>((Pc = X ~> Pc = Y) 
/\ ~((Pc = X ~> Pc = Z ~> Pc = Y) /\ (Pc = Y ~> Pc = X)))

We find that the above formula is equivelent to equation 2 when Y = X

# TODO

- SASS parser
- Verify this code generator is correct (especially branches). Write a small
  register program
- Mutate SASS to verify tool can detect incorrect programs (delete one of the
  reg alloc instructions)

- Literature review