import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import math


def draw(objects):
    x_min = min([obj.min_x for obj in objects])
    y_min = min([obj.min_y for obj in objects])
    x_max = max([obj.max_x for obj in objects])
    y_max = max([obj.max_y for obj in objects])
    dx = x_max - x_min
    dy = y_max - y_min
    plt.figure(figsize=[dx, dy])
    for obj in objects:
        obj.draw()
    plt.xlim(x_min - 0.1 * dx, x_max + 0.1 * dx)
    plt.ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)
    plt.axis('off')


def circle_points(x_0, y_0, radius, n):
    phi = np.linspace(0, 2.0 * np.pi, n)
    x = radius * np.cos(phi) + x_0
    y = radius * np.sin(phi) + y_0
    return x, y


def linear_points(x_0, y_0, x_1, y_1, n):
    x = np.linspace(x_0, x_1, n)
    y = np.linspace(y_0, y_1, n)
    return x, y


class Circle:
    def __init__(self, x, y, width=1, color='w'):
        self.x = x
        self.y = y
        self.width = width
        self.color = color
        self.min_x = x - width
        self.max_x = x + width
        self.min_y = y - width
        self.max_y = y + width

    def draw(self):
        x, y = circle_points(self.x, self.y, self.width, 100)
        plt.plot(x, y, 'k')
        plt.fill_between(x[:50], y[:50], y[99:49:-1], color=self.color)


def bipartite_graph(data, clusters, reorder=None, nu=0.0, mu=1.0, dotscale=1.0):
    if reorder is None:
        reorder = range(len(clusters))

    from_names = dict(enumerate(data.index.values))
    to_names = dict(enumerate(data.columns.values))

    objects = []
    dy = 0.4
    dfromy = dy * (len(from_names) / 2)
    dtoy = dy * (len(to_names) / 2)
    xfrom, yfrom = linear_points(0, -dfromy, 0, dfromy, len(from_names))
    xto, yto = linear_points(10, -dtoy, 10, dtoy, len(to_names))

    cmaps = [plt.cm.Reds, plt.cm.Purples, plt.cm.Greens]
    cmin = np.min(data.values)
    cmax = np.max(data.values)
    cNorm = colors.Normalize(vmin=cmin, vmax=cmax)
    scalarMaps = [cmx.ScalarMappable(norm=cNorm, cmap=cmap) for cmap in cmaps]

    dotweight = cmin + dotscale * (cmax - cmin)

    radius = 0.1
    from_indices, to_indices = np.unravel_index(np.argsort(data.values.ravel()),
                                                data.shape)
    for from_index, to_index in zip(from_indices, to_indices):
        if from_index != to_index:
            x_0 = xfrom[reorder[from_index]]
            y_0 = yfrom[reorder[from_index]]
            x_1 = xto[reorder[to_index]]
            y_1 = yto[reorder[to_index]]
            dx = x_1 - x_0
            dy = y_1 - y_0
            l = math.sqrt(dx * dx + dy * dy)
            x_0 = x_0 + radius * (dx / l)
            y_0 = y_0 + radius * (dy / l)
            x_1 = x_1 - radius * (dx / l)
            y_1 = y_1 - radius * (dy / l)
            from_name = from_names[from_index]
            to_name = to_names[to_index]
            weight = data[to_name][from_name]
            from_cluster = clusters[from_index]
            colorVal = list(scalarMaps[from_cluster].to_rgba(weight))
            colorVal[-1] = nu + (mu - nu) * (weight - cmin) / (cmax - cmin)
            objects.append(Line(x_0, y_0, x_1, y_1, color=colorVal))

    for i in range(len(from_names)):
        x = xfrom[reorder[i]]
        y = yfrom[reorder[i]]
        label = from_names[i]
        objects.append(Circle(x, y, width=radius,
                              color=scalarMaps[clusters[i]].to_rgba(dotweight)))
        objects.append(
            Label(x - 2 * radius, y, label, horizontalalignment='right'))

    for i in range(len(to_names)):
        x = xto[reorder[i]]
        y = yto[reorder[i]]
        label = to_names[i]
        objects.append(Circle(x, y, width=radius,
                              color=scalarMaps[clusters[i]].to_rgba(dotweight)))
        objects.append(
            Label(x + 2 * radius, y, label, horizontalalignment='left'))

    draw(objects)

class Line:
    def __init__(self, x_0, y_0, x_1, y_1, linewidth=1, color='k'):
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_1 = x_1
        self.y_1 = y_1
        self.linewidth = linewidth
        self.color = color
        self.min_x = min(x_0, x_1)
        self.min_y = min(y_0, y_1)
        self.max_x = max(x_0, x_1)
        self.max_y = max(y_0, y_1)

    def draw(self):
        plt.plot([self.x_0, self.x_1], [self.y_0, self.y_1],
                 linewidth=self.linewidth, color=self.color)


class Label:
    def __init__(self,
                 x,
                 y,
                 s,
                 horizontalalignment='center',
                 verticalalignment='center'):
        self.x = x
        self.y = y
        self.s = s
        self.min_x = x
        self.max_x = x
        self.min_y = y
        self.max_y = y
        self.horizontalalignment = horizontalalignment
        self.verticalalignment = verticalalignment

    def draw(self):
        plt.text(self.x, self.y, self.s,
                 horizontalalignment=self.horizontalalignment,
                 verticalalignment=self.verticalalignment)


class OffsetArrow:
    def __init__(self, x_0, y_0, x_1, y_1, crop, linewidth=1.0, color='k'):
        self.dx = x_1 - x_0
        self.dy = y_1 - y_0
        d = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        self.x_0 = x_0 + crop * self.dx / d - self.dy / (10 * d)
        self.y_0 = y_0 + crop * self.dy / d + self.dx / (10 * d)
        self.x_1 = x_1 - crop * self.dx / d - self.dy / (10 * d)
        self.y_1 = y_1 - crop * self.dy / d + self.dx / (10 * d)
        self.dx1 = self.x_1 - self.x_0
        self.dy1 = self.y_1 - self.y_0
        self.linewidth = linewidth
        self.color = color
        self.min_x = min(x_0, x_1)
        self.min_y = min(y_0, y_1)
        self.max_x = max(x_0, x_1)
        self.max_y = max(y_0, y_1)

    def draw(self):
        plt.arrow(self.x_0,
                  self.y_0,
                  self.dx1,
                  self.dy1,
                  head_width=0.1,
                  head_length=0.1,
                  linewidth=self.linewidth, color=self.color)
