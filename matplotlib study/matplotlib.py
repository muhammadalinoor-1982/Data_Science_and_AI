# ****************************  Matplotlib Study **************************

# Link: https://www.youtube.com/playlist?list=PLjVLYmrlmjGcC0B_FP3bkJ-JIPkV5GuZR

# *********** Plot: linear, scatter, bar, stem, step, hist, box, pie, fill between/area plot  ***********

# Install Jupyter Note Book: Go to Python Installation Folder then Go to Script Folder 
# Then Type 'cmd' in the Address bar. Then Type "pip install jupyter" in command promt
# Open Jupyter Notebook: Type "jupyter notebook" in the same command promt
# Open a file in the jupyter notebook: Go to New and Select Notebook then Select python3 (Kernel)
# Then Type "pip install matplotlib" in command promt
# Use matplotlib in 2 ways: 'from matplotlib import pyplot as plt' OR 'import matplotlib.pyplot as plt' 

# Function of Plot are: 
'''
plot(), 
show(),
bar(),
scatter(),
hist(), 
pie(), 
boxplot(), 
stackplot(), 
step(), 
fill_between(), 
subplot(), 
savefig(), 
xticks(), 
yticks(), 
xlim(), 
ylim(), 
axis(), 
text(), 
annotate(), 
legend()
'''
import matplotlib.pyplot as plt
# Show symple Linear plot using plot() and show() Function
x = [1,3,5,7,9]
y = [2,4,6,8,10]
plt.plot(x,y)
plt.show()

# Show symple bar plot using bar() Function with color attribute 
x = [1,3,5,7,9]
y = [2,4,6,8,10]
c = ['r', 'b', 'g', 'y', 'r']
plt.bar(x,y, color=c)
plt.show()

# Show symple bar plot using bar() Function with verious perameter 
import matplotlib.pyplot as plt
import numpy as np

x = ['lichi', 'painapple', 'banana', 'mango', 'orange', 'watermelon']
y = [75, 25, 35, 80, 60, 55]
w = [78, 35, 30, 85, 65, 50]
z = ['orange', 'brown', 'r', 'b', 'black', 'g']

width=0.1
p = np.arange(len(x))
p1 = [j + width for j in p] # convert x list to p array 

plt.xlabel('Frutes', fontsize=20)
plt.ylabel('popularity')
plt.title('Fruites Statistics')

plt.bar(p,y, width=0.1, color=z, align='edge', edgecolor='y', linewidth=2, linestyle=':', alpha=0.99, label='AA') # for horizontal bar use barh
plt.bar(p1,w, width=0.1, color=z, align='edge', edgecolor='r', linewidth=2, linestyle=':', alpha=0.50, label='BB')

plt.xticks(p+width/2, x, rotation=20) # Show the name of x value
plt.legend()
plt.show()

# Show scatter plot using scatter() Function with verious perameter 
x = [5,1,6,4,9,4,2,2,3,4]
y = [9,8,3,3,1,6,1,4,1,8]
z = [6,4,2,6,4,2,4,3,4,8]
size = [99,89,39,39,19,69,19,49,19,89]
color = [46,98,24,42,63,51,34,85,74,98]

plt.scatter(x,y, c=color, s=size, marker='^', cmap='viridis')
plt.scatter(z,y, color='b')

t=plt.colorbar()
t.set_label('Color Bar')
plt.title('scatter plot', fontsize=20)
plt.xlabel('day')
plt.ylabel('number')

plt.show()

# Show Histogram plot using hist() Function with verious perameter 
no = [10, 39, 28, 26, 35, 55, 21, 39, 27, 16, 25, 15, 24, 51, 28, 15, 18, 54, 15, 45, 48, 48, 56, 29,
 55, 48, 46, 51, 48, 58, 54, 33, 37, 55, 45, 32, 24, 39, 33, 40, 55, 52, 50, 42, 25, 58, 45, 10,
 46, 10]

l = [10,20,30,40,50,60] # Use for bins=l, perameter

plt.hist(no, 'auto', (0, 100), edgecolor='black', cumulative=-1, bottom=10, align='mid', histtype='barstacked', orientation='vertical', rwidth=0.8, log=True, label='Test')

plt.title('Histogram plot', fontsize=20)
plt.xlabel('day')
plt.ylabel('number')
plt.axvline(25, color='g', label='Green Line')
plt.legend()
plt.grid()

plt.show()

# Show PIE Chart plot using pie() Function with verious perameter
import matplotlib.pyplot as plt
x = [75, 25, 35, 80, 60, 55] 
y = ['lichi', 'painapple', 'banana', 'mango', 'orange', 'watermelon']
ex = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
c = ['r', 'b', 'g', 'y', 'orange', 'black']

plt.pie(x, 
        labels=y, 
        explode=ex, 
        colors=c, 
        autopct='%0.2f%%', 
        shadow=True, 
        radius=1.1,
        labeldistance=1.1,
        startangle=90,
        textprops={'fontsize':15},
        counterclock=True,
        wedgeprops={'linewidth':6, 'edgecolor':'lightgreen'},
        center=(0.1,0.1),
        rotatelabels=True
       )

plt.title('PIE Chart')
plt.legend(loc=4)

plt.show() 

# Show PIE Chart plot using pie() Function with verious perameter
x = [75, 25, 35, 80, 60, 55]
y = [78, 35, 30, 85, 65, 50]

plt.stem(x,y, linefmt=':', markerfmt='ro', bottom=0, basefmt='b', label='python', orientation='horizontal' ) # use_line_collection=False
plt.legend()
plt.show()

# Show Box Plot and Whisker Plot using boxplot() Function with verious perameter
import matplotlib.pyplot as plt

x = [10,20,30,40,50,60,70,120]
y = [42,66,24,56,43,53,62,44,36,44,160]
z = [x, y]

plt.boxplot(z, 
            labels=['X Box Plot', 'Y Box Plot'], 
            patch_artist=True, 
            showmeans=True, 
            whis=1.5, 
            sym='r*',
            boxprops=dict(color='black'),
            capprops=dict(color='g'),
            whiskerprops=dict(color='b'),
            flierprops=dict(markeredgecolor='magenta')
           ) # notch=True, vert=False, widths=0.5

plt.show()

# Show Stack Plot and Area Plot using stackplot() Function with verious perameter
import matplotlib.pyplot as plt

w = [10,20,30,40,50,60,70,120]
x = [42,66,24,56,43,53,62,44]
y = [72,86,93,22,31,16,71,32]
z = [89,16,54,67,81,23,16,48]
l = ['Dhaka', 'Chittagong', 'Khulna']

plt.stackplot(w,x,y,z, labels=l, colors=['r', 'g', 'b'], baseline='zero')
# baseline='sym','zero','wiggle' 

plt.title('Stack Plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid()

plt.legend()
plt.show()

# Step Plot using step() Function with verious perameter
import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9]
y = [11,22,33,44,55,66,77,88,99]

plt.step(x,y, color='r', marker='*', ms=10, mfc='b', label='Step Plot')

plt.title('Step Plot')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.grid()
plt.show()

# Fill Between Plot using fill_between() Function with verious perameter
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9]) 
y = np.array([11,22,33,44,55,66,77,88,99])

plt.plot(x, y, color='r')
plt.fill_between(x, y, label='Pick Point', color='g', where=(x>=3)&(x<=7)) #x = [3,8], y1 = 22, y2 = 66,

plt.title('Fill Between Color')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.grid()
plt.legend()

plt.show()

# Sub Plot using subplot() Function with verious perameter
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9]) 
y = np.array([11,22,33,44,55,66,77,88,99])
plt.subplot(2,2,4)
plt.step(x,y, color='r')

plt.pie([1], colors='g')
plt.subplot(2,2,2)

z = [62,84,14,33,48,49]
plt.subplot(2,2,3)
plt.pie(z)

a = ['a', 'b', 'c', 'd', 'e', ]
b = [24,33,24,43,14]
plt.subplot(2,2,1)
plt.bar(a,b)

plt.show()

# Save Figure using savefig() Function with verious perameter
import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9] 
y = [11,22,33,44,55,66,77,88,99]

plt.bar(x,y, color='r')

plt.savefig('Bar_Plot_inch', dpi=100, facecolor='b', transparent=True, bbox_inches='tight')

plt.show()

# xticks(), yticks(), xlim(), ylim() and axis() Function with verious perameter
import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9] 
y = [6,8,3,1,3,6,5,5,3]

plt.plot(x,y)

#plt.xticks(x, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] )
#plt.yticks(x)

#plt.xlim(0.10)
#plt.ylim(0.5)

plt.axis([0,10,0,5])

plt.show()

# text() and annotate() and legend() Function with verious perameter
import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9] 
y = [6,8,3,1,3,6,5,5,3]

plt.plot(x,y)

plt.text(6,6,'Text Function', fontsize=10, style='italic', color='w', bbox={'facecolor':'r'})
plt.annotate('An Notate', xy=(3,3), xytext=(5,7), arrowprops=dict(facecolor='b', shrink=10), color='w', bbox={'facecolor':'r'})
plt.legend(['ok'], loc=9, facecolor='b', edgecolor='r', framealpha=0.3, shadow=True)

plt.show()