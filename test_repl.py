from autoformalization_typechecking.lean_repl import RobustLeanServer

if __name__ == '__main__':
    lean_server = RobustLeanServer()
    print(lean_server.run_code("import Mathlib\ntheorem test : 1 + 1 = 2 := sorry"))  

