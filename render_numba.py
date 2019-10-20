#!/usr/bin/env python3
# import scipy
from numba import jit
import numpy as np

dif = 1e-6
DIST_LIMIT = 1e-3
COUNT_LIMIT = 100
MAX_DIST = 30

deltas = np.eye(3)

@jit(nopython=True)
def normal(p: np.ndarray, DE, de: np.double, dif: np.double=1e-6) -> np.ndarray:
    # normal vector to surface
    n = np.zeros(3)
    for i in range(3):
        d = deltas[i]
        q = p+d*dif
        n[i], _, _ = DE(q)
        
    n = (n-de)/dif
    leng = np.linalg.norm(n)
    n = n/leng
    return n
    
@jit(nopython=True)
def shadow(p, normal, light, DE, light_spread=0.001) -> float:
    if (normal*light).sum() < 0.:
        return 0.
    # need to start away from the surfice we are on
    q = p + normal*1*DIST_LIMIT

    de, _, _ = DE(q)
    away_dist = 0.
    min_de = 0.
    min_de_achieved = False
    for counter in range(COUNT_LIMIT):
        if not (de > DIST_LIMIT and away_dist < MAX_DIST):
            break

        away_dist += de
        q = q + light*de
        de, _, _ = DE(q)

        if de < light_spread*away_dist:
            if min_de_achieved:
                min_de = min(min_de, de)
            else:
                min_de = de
                min_de_achieved = True
    
    if de < DIST_LIMIT:
        # when hit an eclipsing light object
        return 0.
        
    if not min_de_achieved:
        return 1.
    # otherwise return oclussion by nearby edges
    return min(1., (min_de/light_spread))

@jit(nopython=True)
def march(origin, direction, DE, light: np.ndarray) -> np.ndarray:
    """
    :params: origin
    :params: direction
    :params DE function: DE  # this function must take a (3,)np.ndarray and return a float
    :params (3,)np.ndarray: light  # the direction from where the ambient light comes
    """
    p = origin
    v = direction
    away_dist = 0.

    de, col, nor = DE(p)
    dist_limit = False
    for counter in range(COUNT_LIMIT):
        if not (de > DIST_LIMIT and away_dist < MAX_DIST):
            dist_limit = True
            break
        p = p + v*de
        away_dist = away_dist+de
        de, col, nor = DE(p)
        
    if de > DIST_LIMIT:
        return np.zeros(3)

    n = normal(p, DE, de) if nor is None else nor
    m = np.absolute(n) if col is None else col

    # light incidence shading
    shade = n*light
    s = min(max(shade.sum(),0.),1.)

    # shadow by eclipsing
    shadowed = shadow(p, n, light, DE, light_spread=.1)

    return m*(.5+.5*(s+shadowed))

def smooth(image):
    new_image = image[:-1,:-1,:] + image[:-1,1:,:] + image[1:,:-1,:] + image[1:,1:,:]
    return new_image/4.

@jit(nopython=True)
def my_cross(a, b):
    c = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    return np.array(c, np.double)
    
@jit(nopython=True)
def v(i, j, main_axe, x, y):
    v_ = main_axe + i*x + j*y
    len_v_ = np.linalg.norm(v_)
    v_unit = v_/len_v_
    return v_unit

@jit(nopython=True) # 'array(float64, 1d, C)(array(float64, 1d, C), array(float64, 1d, C), (int64, int64), float64)', 
def screen(origin, target, shape=(80,80), diag=1.):
    """
    :type origin: np.ndArray() shape=(3,)
    :type target: np.ndArray() shape=(3,)
    :type shape: Tuple
    :type diag: float
    :return type: np.ndArray() shape=(shape[0]*shape[1],3)
    """
    main_axe = target-origin
    shape_arr = np.array(shape, dtype=np.double)
    shape_arr_len = np.linalg.norm(shape_arr)
    unit = diag/shape_arr_len
    z = np.array((0.,0.,1.))
    horiz_ = my_cross(main_axe, z)
    verti_ = my_cross(horiz_, main_axe)
    # print('horiz {} verti_ {}'.format(horiz_, verti_))
    len_horiz = np.linalg.norm(horiz_)
    len_verti = np.linalg.norm(verti_)
    x = unit*horiz_/len_horiz
    y = unit*verti_/len_verti
    yr, xr = shape
    x0, y0 = xr/2, yr/2
    iterable = [(-x0+n%xr, y0-n//xr) for n in range(xr*yr)]
    scr_ = np.zeros((xr*yr, 3), np.double)
    for n, (i, j) in enumerate(iterable):
        scr_[n, :] = v(i,j, main_axe, x, y)
    return scr_

@jit(nopython=True)
def render2(origin, target, DE, screen, light, shape=(80,80)):

    # direction is normalized, comes from origin and points to the target
    direction = target - origin
    dir = direction/np.linalg.norm(direction)

      # march performs the render for every pixel in the screen
    image = np.zeros((screen.shape[0],3))
    for i in range(screen.shape[0]):
        dir = screen[i, :]
        # print('dir: ', dir)
        image[i, :] = march(origin,dir,DE,light)

    print('render 2 finished')
    image = image.reshape(*shape,3)
    return image

class SphereDE():
    def __init__(self, radius=1., pos=np.array((0,0,0))):
        self.radius = radius
        self.pos = pos
    @jit(nopython=True)
    def __call__(self, P: np.ndarray) -> float:
        rod = P - self.pos
        dist = np.linalg.norm(rod)
        de = dist - self.radius
        color = np.array([abs(rod[i])/dist<0.8 for i in range(3)], dtype=np.double)
        normal = rod/dist
        return de, color, normal

class PlaneDE():
    def __init__(self, normal=np.array((0,0,1)), pos=np.array((0,0,0))):
        self.pos = pos
        self.normal = normal/np.linalg.norm(normal)
    @jit(nopython=True)
    def __call__(self, p: np.ndarray) -> float:
        rod = p - self.pos
        de = (self.normal*rod).sum()
        normal = self.normal 
        if de < 0:
            de, normal = -de, -normal
        color = np.array((0,0,1), dtype=np.double)
        return de, color, normal

class ComposeDE():
    def __init__(self, bodys=[]):
        self.bodys = bodys # each element in bodys must by a DE function    
    @jit(nopython=True)
    def __call__(self, p: np.ndarray) -> float:
        # des = np.array([DE(p) for DE in self.bodys])
        # return des.min()
        des_ = [DE.__call__(p) for DE in self.bodys]
        des, colors, normals = zip(*des_)
        min_de = np.argmin(des)
        return des[min_de], colors[min_de], normals[min_de]

if __name__ == "__main__":

    moovie = True

    import matplotlib.pyplot as plt    
    from matplotlib.animation import FuncAnimation
    from os import rename

    import time
    s = time.perf_counter()

    sph_DE = SphereDE(radius=.5, pos=np.zeros((3,), np.double))
    sph2_DE = SphereDE(radius=.3, pos=np.zeros((3,), np.double))
    pln_DE = PlaneDE(normal=np.array((0, .1, 1.)), pos=np.array((0,0,-1)))
    comp_DE = ComposeDE([sph_DE, sph2_DE, pln_DE])

    # @jit(nopython=True)
    # def fase_DE(p, fase):
    #     ax1 = np.array((.4, -1., .5))*np.sin(fase)
    #     ax2 = np.array((.6, .6, .1))*np.cos(fase)
    #     centre = np.array((.0, .2, .0))
    #     satelite = centre + ax1 + ax2
    #     sph_DE.pos = centre
    #     sph2_DE.pos = satelite
    #     return comp_DE.__call__

    @jit(nopython=True)
    def fase_DE(p, fase):
        ax1 = np.array((.4, -1., .7))*np.sin(fase)
        ax2 = np.array((1.2, 1.2, .1))*np.cos(fase)
        centre = np.array((.0, .2, .0))
        satelite = centre + ax1 + ax2
        plano = np.array((0, .1, 1.))
        a = np.linalg.norm(p - centre) - .5
        b = np.linalg.norm(p - satelite) - .3
        c = ((p-np.array((0,0,-1.5)))*plano).sum()
        min_obj = np.argmin(np.array((a,b,c)))
        switcher = {
            0: (a,
                ((p-centre)/(a+.5))**2,
                (p-centre)/(a+.5)),
            1: (b,
                1.-((p-satelite)/(b+.3))**2,
                (p-satelite)/(b+.3))
        }
        default = (c,
                   np.ones((3,), np.double)*(int(p[0]*2)%2==int(p[1]*2)%2),
                   plano)
        return switcher.get(min_obj, default)

    origin = np.array((2.,-1.,5.))
    target = np.array((0.,0.,-2.))
    light = np.array((1.,1.,1.))
    eye = np.array((.2,.0,.0))

    shape = (200, 180)

    # light needs to be normalized
    light_len = np.linalg.norm(light)
    light = light/light_len
    scrL = screen(origin+eye, target, shape, diag=8.)
    scrR = screen(origin-eye, target, shape, diag=8.)

    def new_image(fase):
        # fase = -.2

        @jit(nopython=True)
        def comp_DE(p):
            return fase_DE(p, fase)

        kws = dict(origin=origin+eye, target=target, 
                DE=comp_DE, screen=scrL,
                light=light, 
                shape=shape)
        imageL = render2(**kws)

        kws = dict(origin=origin-eye, target=target, 
                DE=comp_DE, screen=scrR,
                light=light, 
                shape=shape)
        imageR = render2(**kws)

        image = np.concatenate([imageL,imageR], axis=1)
        return image


    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(np.zeros((shape[0],shape[1]*2)))

    if moovie:
        # f = time.perf_counter()
        # elapsed = f - s
        # print(f"{__file__} executed in {elapsed:0.6f} seconds.")

        def update(i):
            im.set_data(new_image(i))
            return im,

        file = 'images/'+str(time.time())+'.gif'
        # p = time.perf_counter()
        fnAn = FuncAnimation(fig, update, frames=np.linspace(np.pi/6,2*np.pi,12), interval=.02, blit=True, repeat=True)
        fnAn.save(file, dpi=80, writer='imagemagick')
        # elapsed = int(time.perf_counter() - p)
        # rename(file, f'numbatest_eta{elapsed:d}s.gif')

    else:

        plt.imshow(new_image(.6))

        plt.show()
