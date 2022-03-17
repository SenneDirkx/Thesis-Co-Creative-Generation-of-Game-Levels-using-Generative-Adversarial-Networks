from distutils.command.config import config
from sympy import *
from typing import List

# CONSTRAINTS
# 1) minimally 1 door: 2 blocks left or right, 3 blocks top or bottom
# 2) outer ring always walls 
# 3) inner room never walls

# minimally 1 door
door_left, door_right, door_top, door_bottom = symbols('door_left,door_right,door_top,door_bottom')
doors = door_left | door_right | door_top | door_bottom

# outer ring always walls
wall_left, wall_right, wall_top, wall_bottom = symbols('wall_left,wall_right,wall_top,wall_bottom')
outer_walls = wall_left & wall_right & wall_top & wall_bottom

# inner room never walls
c2_wall, c3_wall, c4_wall, c5_wall, c6_wall, c7_wall, c9_wall, c10_wall, c11_wall, c12_wall, c13_wall = symbols('c2_wall,c3_wall,c4_wall,c5_wall,c6_wall,c7_wall,c9_wall,c10_wall,c11_wall,c12_wall,c13_wall')
d2_wall, d3_wall, d4_wall, d5_wall, d6_wall, d7_wall, d9_wall, d10_wall, d11_wall, d12_wall, d13_wall = symbols('d2_wall,d3_wall,d4_wall,d5_wall,d6_wall,d7_wall,d9_wall,d10_wall,d11_wall,d12_wall,d13_wall')
e2_wall, e3_wall, e4_wall, e5_wall, e6_wall, e7_wall, e9_wall, e10_wall, e11_wall, e12_wall, e13_wall = symbols('e2_wall,e3_wall,e4_wall,e5_wall,e6_wall,e7_wall,e9_wall,e10_wall,e11_wall,e12_wall,e13_wall')
f2_wall, f3_wall, f4_wall, f5_wall, f6_wall, f7_wall, f9_wall, f10_wall, f11_wall, f12_wall, f13_wall = symbols('f2_wall,f3_wall,f4_wall,f5_wall,f6_wall,f7_wall,f9_wall,f10_wall,f11_wall,f12_wall,f13_wall')
g2_wall, g3_wall, g4_wall, g5_wall, g6_wall, g7_wall, g9_wall, g10_wall, g11_wall, g12_wall, g13_wall = symbols('g2_wall,g3_wall,g4_wall,g5_wall,g6_wall,g7_wall,g9_wall,g10_wall,g11_wall,g12_wall,g13_wall')
h2_wall, h3_wall, h4_wall, h5_wall, h6_wall, h7_wall, h9_wall, h10_wall, h11_wall, h12_wall, h13_wall = symbols('h2_wall,h3_wall,h4_wall,h5_wall,h6_wall,h7_wall,h9_wall,h10_wall,h11_wall,h12_wall,h13_wall')
i2_wall, i3_wall, i4_wall, i5_wall, i6_wall, i7_wall, i9_wall, i10_wall, i11_wall, i12_wall, i13_wall = symbols('i2_wall,i3_wall,i4_wall,i5_wall,i6_wall,i7_wall,i9_wall,i10_wall,i11_wall,i12_wall,i13_wall')

room_no_walls_c = ~c2_wall & ~c3_wall & ~c4_wall & ~c5_wall & ~c6_wall & ~c7_wall & ~c9_wall & ~c10_wall & ~c11_wall & ~c12_wall & ~c13_wall
room_no_walls_d = ~d2_wall & ~d3_wall & ~d4_wall & ~d5_wall & ~d6_wall & ~d7_wall & ~d9_wall & ~d10_wall & ~d11_wall & ~d12_wall & ~d13_wall
room_no_walls_e = ~e2_wall & ~e3_wall & ~e4_wall & ~e5_wall & ~e6_wall & ~e7_wall & ~e9_wall & ~e10_wall & ~e11_wall & ~e12_wall & ~e13_wall
room_no_walls_f = ~f2_wall & ~f3_wall & ~f4_wall & ~f5_wall & ~f6_wall & ~f7_wall & ~f9_wall & ~f10_wall & ~f11_wall & ~f12_wall & ~f13_wall
room_no_walls_g = ~g2_wall & ~g3_wall & ~g4_wall & ~g5_wall & ~g6_wall & ~g7_wall & ~g9_wall & ~g10_wall & ~g11_wall & ~g12_wall & ~g13_wall
room_no_walls_h = ~h2_wall & ~h3_wall & ~h4_wall & ~h5_wall & ~h6_wall & ~h7_wall & ~h9_wall & ~h10_wall & ~h11_wall & ~h12_wall & ~h13_wall
room_no_walls_i = ~i2_wall & ~i3_wall & ~i4_wall & ~i5_wall & ~i6_wall & ~i7_wall & ~i9_wall & ~i10_wall & ~i11_wall & ~i12_wall & ~i13_wall

room_no_walls = room_no_walls_c & room_no_walls_d & room_no_walls_e & room_no_walls_f & room_no_walls_g & room_no_walls_h & room_no_walls_i

total = doors & outer_walls #& room_no_walls
cnf = to_cnf(total, simplify=False, force=True)

def to_dimacs(cnf: str) -> List[str]:
    count = 1
    clauses = []
    variables = {}
    splitted = cnf.strip().split(" & ")

    for clause in splitted:
        result = ""
        if clause[0] == '(':
            or_clause = clause[1:-1].strip(" ").split(" | ")
            for literal in or_clause:
                if literal in variables:
                    number = variables[literal]
                else:
                    number = count
                    count += 1
                    variables[literal] = number
                result += str(number) + ' '
            result += '0'
        elif clause[0] == '~':
            literal = clause[1:]
            if literal in variables:
                number = variables[literal]
            else:
                number = count
                count += 1
                variables[literal] = number
            result += str(number) + ' 0'
        else:
            literal = clause
            if literal in variables:
                number = variables[literal]
            else:
                number = count
                count += 1
                variables[literal] = number
            result += str(number) + ' 0'


        clauses.append(result)
    
    
    return [f"p cnf {count-1} {len(clauses)}"] + clauses

dimacs = to_dimacs(str(cnf))
dimacs_str = "\n".join(dimacs)
with open("./basic_constraint.txt", 'w') as constraints_file:
    constraints_file.write(dimacs_str)