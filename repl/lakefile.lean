import Lake
open Lake DSL

package REPL {
  -- add package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib REPL {
  -- add library configuration options here
}

@[default_target]
lean_exe repl where
  root := `REPL.Main
  supportInterpreter := true
