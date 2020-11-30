import numpy

gap_dict = {0:6, 1:4, 2:2, 3:2, 4:1, 5:1,
                 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0,
                 14:1, 15:1, 16:2, 17:2, 18:4, 19:6}

board = numpy.zeros((20, 20), dtype="int32")    

tile_dict = {}  
tile_index = 0
for row in range(20):
    column = 0
    while column < 20 - gap_dict[row]:
        if column == 0 and gap_dict[row] != 0:
            column += gap_dict[row]
        board[column][row] = 1
        tile_dict[tile_index] = [column, row]
        tile_index += 1
        column += 1
board -= 1 

cost_array = numpy.copy(board)
with open("tileCost.xml", "r") as cost_file:
    lines = cost_file.readlines()
    read = False
    ind = 0
    for line in lines:
        if "</Item>" in line:
            read = False
        if read == True:
            cost_array[tile_dict[ind][0]][tile_dict[ind][1]] = line.replace("\t", "").replace("\n", "")
            ind += 1  
        if "<Item>" in line:
            print("hmmmm")
            read = True


print(cost_array)