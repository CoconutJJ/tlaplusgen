#import "slides_template.typ": *
#show: conf
#import "@preview/cetz:0.4.2": canvas, draw
#set document(
  author: "David Yue, Leo Sciortino",
  date: datetime(year: 2026, month: 2, day: 23),
  title: "CS6969: Fast and Verified GPU: Final Project Presentation",
  
)
#let Accent = rgb("#76B900")
#let Dark = rgb("#222222")
#set text(fill: Dark)


// ----------------------------
// Title slide
// ----------------------------
#title-slide(
  [Verification of the `USETMAXNREG` instruction on NVIDIA GPUs],
  "CS6969: Fast and Verified GPU: Final Project Presentation",
  "Presented By: David Yue, Leo Sciortino",
)

// ----------------------------
// Content
// ----------------------------
#body-frame("Background", [
  - To achieve peak performance on modern Nvidia GPUs warp specialized is used
    - Commonly there is a producer warp loading via TMA and a consumer warp that is performing arithmetic in tensor cores 
    - The `usetmaxnreg` instruction allows threads in a warp group to collectively increase or decrease the number of registers that are in use.
    - Specifically TMA producer warps generally decrement their register counts giving their registers to consumer warps
    - The register increase and decrease instructions take a register number, once the call is made, the warp group is no longer allowed to use registers above that number.

])

#body-frame("TLA+", subtitle:"Background", [
  - Formal specification language designed by Leslie Lamport for describing and reasoning about concurrent and distributed systems at a high level of abstraction, above code.
  - Based on temporal logic of actions — systems are modeled as state machines, with behaviors expressed using logical operators over states and state transitions (e.g., [] always, <> eventually, ~> leads to).
  
  - Checks properties, not just tests them — the TLC model checker exhaustively explores the state space to verify safety (nothing bad happens) and liveness (something good eventually happens) properties.
])
#body-frame("Goal of the project: Correctness", [
  - Verify the correctness of NVIDIA GPU's dynamic register allocation instructions.
    - Make every request from a warp group to increase the register count will be eventually be fulfilled (i.e another warp group will decrease it's register count, or there is already enough in the pool)
    - Make sure that every thread only uses register numbers that are below the max register number previously specified by increase or decrease.
])

#body-frame("Division of Work", [
  - Parser, Slicer - Using Sree's tools and some help from Claude Code (mostly Prof. Sreepathi Pai)
  - TLA Codegen python API - Mostly David, Leo added some features when needed 
  - Semantics - Sree provided us with most semantics. Leo dervied some using Sree's NVBit tool (mostly human)
  - Invariants - Mostly Leo 

  Leo and David worked together on most of the project so the division isn't a hard line

])
#body-frame("Invariants")[
  1. Guarantee that before calling usetmaxnreg again there is a bar.sync
  2. Guarantee that argument to `usetmaxnreg` doesn't go below 24 or above 255, and that it is a multiple of 8
  3. Every thread that incs their registers must eventually not be in that state
  4. If one thread in a warp group incs or decs its registers, then all threads in the warp group must do the same
  5. If an instruction has register $R_i$ as one of it's operands, then the number of registers allocated to that thread must be $>= i+1$
  6. A thread that incs to $n$ registers must have $<= n$ registers currently allocated
  7. A thread that decs to $n$ registers must have $>= n$ registers currently allocated 
The above conditions are implemented through TLA+ invariants, and properties. Some conditions needed extra states.

]
#body-frame("What works?", [
  - We have a great TLA+ code generator
  - We have invariants and properties implemented
  - We have most of the SASS semantics implemented in TLA+
])

#body-frame("What still doesn't work", [
  - Missing or unsure about many SASS instruction semantics - SASS instructions are not documented publicly. Most of what is online - which is very little - has been reverse engineered.
  - Can we remove certain sass instructions that don't control the behavior of `USETMAXNREG`?
  - Parser and slicer aren't fully rigid yet. SASS dump formats do not have any documented grammar.

])

#body-frame("Future Work", [
  - We plan to continue working and submit to POPL 27'
  - Possibly look at a PTX version (right now we are using SASS)
  - Finish deriving semantics and testing more
  - Improve parser and slicer
  - Test with code mutations
  - Formalize the semantics
  - Can we automatically insert `usetmaxnreg` and verify the code with our tool?
])

#body-frame("Credits")[
  - Thank you to Prof. Sreepathi Pai for his advising on this project.
    - He also provided us tools such as SASS parser, slicer, and NVBit tool for collecting register traces: 
    - https://github.com/pyxis-roc/gpucode-analyzer
  - Thanks to Amir (Claude) for his guidance
]

#slide([
  #align(center + horizon)[
    #text(fill: Accent)[#heading("Thank You", depth: 1)]
  ]
])
