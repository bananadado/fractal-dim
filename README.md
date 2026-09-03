# fractal-dim
Box-counting fractal dimension estimator via L-systems.
Writeup by ME, everything else pretty clauded with a lot of guidance.

# writeup
This is just some findings which I thought were interesting as well as a not-so-shallow dive into the actual maths behind it which is pretty interesting.
Actually took me forever to get my head round the Lebesgue measure theory.

The first part of the project drawing L-systems is loosely based on a year 1 haskell homework I did as part of my course. The rest just expands on it.

# running the code

From the project root, with the environment set up (see below):

```sh
python3 -m fractaldim list
python3 -m fractaldim list --full                 # also show the productions
python3 -m fractaldim draw hilbert -n 5
python3 -m fractaldim draw plant -n 5 --open
python3 -m fractaldim animate hilbert -n 4 --open # watch the turtle draw it
```

## the virtualenv

There is no environment in the repository so build yourself with uv.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

`-e` installs in editable mode, so edits to `fractaldim/` take effect with no
reinstall. To use `python3` and `fractaldim` as bare commands, activate the
environment, which puts them on your `PATH` for the rest of the shell session.
This must be **sourced**, not executed -- `activate` is deliberately not an
executable file, so running it directly gives "permission denied":

```sh
source .venv/bin/activate
```

Without activating, spell out the interpreter: `.venv/bin/python -m fractaldim ...`.

Tests: `.venv/bin/python -m pytest`

## commands
There are 5 top level commands: `list`, `growth`, `boxcount`, `draw`, and
`animate`.
Each of these has further options including resolution and fps.


`growth` recovers the dimension from the grammar itself -- the substitution
matrix, the growth rate of the drawn symbols, and the scaling factor -- so
`python3 -m fractaldim growth` prints the write-up's Table 1 as computed
output. Give it a name for the workings.

`boxcount` goes the other way, measuring the dimension off the drawn curve
without using the grammar at all -- an independent check on `growth`. It
prints the count at every box size and the local slope between them, since a
fitted line through box counts always yields a number and only a plateau in
those slopes says the number is a dimension. `--grids` draws the occupied
boxes at four scales.

You can also change the generation of an L-system using `-n` for iterations.

Use `-h` for more information.