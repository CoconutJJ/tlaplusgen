#import "@preview/cetz:0.4.2": canvas, draw

#let Title = black
#let TitleBackground = rgb(247, 247, 247)
#let StatusBar = rgb("#76b900")
#let SlideBg = white
#let TextColor = black

#let handout-mode = state("handout-mode", false)
#let slide-footnotes = state("slide-footnotes", ())
#let slide-step-counter = counter("slide-step-counter")
#let slide-id-counter = counter("slide-id-counter")

#let conf(handout: false, doc) = {
  set page(width: 16in, height: 9in, margin: 0.5in, fill: SlideBg)
  set text(font: "New Computer Modern", size: 28pt, fill: TextColor)
  set heading(bookmarked: false)
  show heading.where(level: 1): set text(size: 44pt)
  show heading.where(level: 2): set text(size: 34pt)
  handout-mode.update(handout)
  doc
}

#let reveal(n, content) = context {
  // Use the unique slide ID to tag this metadata
  let id = slide-id-counter.get().first()
  [#metadata(n) #label("step-" + str(id))]

  if (handout-mode.get()) { return content }

  if (slide-step-counter.get().first() >= n) {
    content
  } else {
    text(fill: gray)[#content]
  }
}

#let only(n, content) = context {
  // Use the unique slide ID to tag this metadata
  let id = slide-id-counter.get().first()
  [#metadata(n) #label("step-" + str(id))]

  if (handout-mode.get()) { return content }

  if (slide-step-counter.get().first() >= n) {
    content
  } else {
    hide(content)
  }
}

#let slide(body) = context {
  slide-id-counter.step()
  slide-step-counter.update(0)

  let id = slide-id-counter.get().first()
  let reveals = query(label("step-" + str(id)))
  let max-n = 0
  for r in reveals {
    if r.value > max-n {
      max-n = r.value
    }
  }

  for s in range(0, max-n + 1) {
    if here().page() >= 1 or s > 0 {
      pagebreak(weak: true)
    }

    slide-step-counter.update(s)

    // reset footnotes after every partial slide, this prevents partial slides from adding duplicate footnotes
    slide-footnotes.update(())

    body
  }
}

#let title-slide(title, subtitle, author, date: datetime.today(), handout: false) = {
  set document(
    author: author,
    date: date,
    title: [#title - #subtitle],
  )
  slide(
    [
      #align(horizon)[
        #pad(left: -0.5in)[
          #block(fill: TitleBackground, inset: 1cm, width: 95%)[
            #if (subtitle.len() > 0) [
              #smallcaps[#text(size: 28pt)[*#subtitle*]]
            ]
            #text(fill: Title)[#heading(title, bookmarked: true, depth: 1)]\
            #text(size: 22pt)[*#author*]\
            #text(size: 22pt)[#date.display("[month repr:long] [day], [year]")]
          ]
        ]
      ]
    ],
  )
}

#let section-slide(content) = slide[
  #align(center + horizon)[
    #content
  ]
]

#let slide-footnote(content) = context {
  let notes = slide-footnotes.get()
  let count = notes.len() + 1
  slide-footnotes.update(n => n + (content,))
  super(text(size: 1em, fill: blue)[#count])
}

#let body-frame(title, subtitle: "", max_steps: 0, body) = slide[
  #block[
    #if subtitle.len() > 0 {
      smallcaps[#text(size: 24pt)[*#subtitle*]]
      v(-0.8em)
    }
    #text(fill: Title)[
      #heading(title, bookmarked: true, depth: 2)
    ]
    #line(length: 100%, stroke: TextColor)
  ]
  #block(height: 1fr)[
    #align[
      #body
    ]
  ]
  #place(bottom)[
    #context {
      let notes = slide-footnotes.get()
      if notes.len() > 0 {
        set text(size: 14pt)
        line(length: 33%, stroke: gray)
        grid(
          columns: 2,
          gutter: 1em,
          ..notes.enumerate().map(((i, n)) => [#super(str(i + 1)) #n])
        )
      }
    }
    #pad(
      [
        #block(width: 100%, fill: StatusBar, inset: 0.4cm, clip: false)[
          #text(stroke: white, fill: white, size: 20pt)[
            #grid(columns: (1fr, auto))[
              #smallcaps({
                if (subtitle.len() > 0) [
                  #subtitle:
                ]
                title
              })
            ][
              #align(right)[
                #context {
                  [#here().page()/#counter(page).final().first()]
                }
              ]
            ]
          ]
        ]
      ],
      bottom: -0.5in,
      left: -0.5in,
      right: -0.5in,
    )
  ]
]

