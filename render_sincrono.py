#!/usr/bin/env python3
import asyncio
import numpy as np


dif = 1e-8
DIST_LIMIT = 1e-6
COUNT_LIMIT = 100
MAX_DIST = 30
MAX_REFLEX = 3
REFLEXIVITY = .4

class SphereDE():
    def __init__(self, radius=1., pos=np.array((0,0,0))):
        self.radius = radius
        self.pos = pos
    def __call__(self, dims: tuple) -> float:
        p, d = dims # "d" stands for direction and should always be a versor.
        separation = self.pos - p
        dist = np.linalg.norm(separation)
        cos_ang = (d*separation).sum()/dist
        # print(f'Sphere:\n\tp=({p}), d=({d}), dist={dist}, cos(ang)={cos_ang}')
        b = dist*cos_ang
        discriminant = b**2-(dist*dist-self.radius*self.radius)
        de = b-(discriminant)**.5 if discriminant >= 0. else float("inf")
        if de < 0:
            return float("inf"), np.zeros((3,)), np.ones((3,))
        elif de < float("inf"):
            new_p = p + d*de
            separation = new_p - self.pos
            dist = np.linalg.norm(separation)
            color = np.array([abs(separation[i])/dist<0.8 for i in range(3)], dtype=np.double)
            normal = separation/dist
        else:
            color = np.zeros((3,))
            normal = np.ones((1,))
        return de, color, normal

class PlaneDE():
    def __init__(self, normal=np.array((0,0,1)), pos=np.array((0,0,0))):
        self.pos = pos
        self.normal = normal/np.linalg.norm(normal)
    def __call__(self, dims: tuple) -> float:
        p, d = dims # "d" stands for direction and should always be a versor.
        
        cos_ang = -(d*self.normal).sum()
        if cos_ang < 0:
            # The ray won't hit the plane
            return float("inf"), np.zeros((3,)), np.ones((3,))
        
        reference = self.pos - p
        de = -(self.normal*reference).sum()/cos_ang
        if de < 0:
            # print(f'Exception!\n\tDistance: {de}\n\torigin: {p}\n\tdirection: {d}\n\tcos(ang): {cos_ang}\n\tReference: {reference}')
            de = 0.

        normal = self.normal
        new_p = p + d*de
        stripe = float(any((new_p%1.)**2 < .0001))
        color = np.array((stripe,stripe,stripe), dtype=np.double)*.8+np.ones(3)*.2
        return de, color, normal

class ComposeDE():
    def __init__(self, bodys=[]):
        self.bodys = bodys # each element in bodys must by a DE function
    def __call__(self, dims: tuple) -> float:
        # des = np.array([DE(p) for DE in self.bodys])
        # return des.min()
        des_ = [DE(dims) for DE in self.bodys]
        des, colors, normals = zip(*des_)
        min_de = np.argmin(des)
        return des[min_de], colors[min_de], normals[min_de]

def ambient_light(normal: np.ndarray, light: np.ndarray) -> float:
    shade = (normal*light).sum()
    return max(0.0, shade)

def shadow2(p, normal, light, DE, light_spread=0.001) -> float:
    q = p# + normal*dif
    de, _, _ = DE((q, light))
    if de < MAX_DIST:
        return 0.
    return 1.

def reflected(d, n):
    d_n = -2*(d*n).sum()*n
    return d + d_n

def march(origin, direction, DE, light: np.ndarray, reflex:int=0) -> np.ndarray:
    """
    :params: origin
    :params: direction
    :params DE function: DE  # this function must take a (3,)np.ndarray and return a float
    :params (3,)np.ndarray: light  # the direction from where the ambient light comes
    """
    de, color, nor = DE((origin, direction))

    if de > MAX_DIST:
        return np.zeros((3,))

    p = origin + direction*de
    n = nor
    shaded = ambient_light(n, light)
    if shaded == 0.:
        shadowed = 0.
    else:
        shadowed = shadow2(p, n, light, DE, light_spread=.2)

    selfcolor = color*(shaded*shadowed*.5 + .5)
    if reflex < MAX_REFLEX:
        v = reflected(direction, nor)
        selfcolor += march(p, v, DE, light, reflex+1)*REFLEXIVITY
    return np.clip(selfcolor, 0., 1.)

class Screen():
    def __init__(self, origin, target, shape=(80,80), diag=1.):
        self.origin = origin
        self.target = target
        self.main_axe = target - origin
        self.shape = shape
        a_shape = np.array(shape, dtype=np.double)
        unit = diag/np.linalg.norm(a_shape)
        horiz_ = np.cross(self.main_axe, np.array((0,0,1)))
        verti_ = np.cross(horiz_, self.main_axe)
        self.x = unit*horiz_/np.linalg.norm(horiz_)
        self.y = unit*verti_/np.linalg.norm(verti_)
    def __iter__(self):
        yr, xr = self.shape
        x0, y0 = xr/2, yr/2
        self.iterable = ((-x0+n%xr, y0-n//xr) for n in range(xr*yr))
        return self
    def __next__(self):
        i, j = next(self.iterable)
        v = self.main_axe + i*self.x + j*self.y
        return v/np.linalg.norm(v)

def smooth(image):
    new_image = image[:-1,:-1,:] + image[:-1,1:,:] + image[1:,:-1,:] + image[1:,1:,:]
    return new_image/4.

def render(screen, DE, light):
    # light needs to be normalized
    light = light.copy()/np.linalg.norm(light)
    # march performs the render for every pixel in the screen
    image = [march(screen.origin,dir,DE,light) for dir in screen]
    print('render finished')
    image = np.array(image).reshape(*screen.shape,3)
    return smooth(image)

def stereo(kwargsL, kwargsR):
    return render(**kwargsL), render(**kwargsR)

if __name__ == "__main__":
    import time
    s = time.perf_counter()

    sph_DE = SphereDE(radius=.5, pos=np.array((.0, .4, .0)))
    sph2_DE = SphereDE(radius=.3, pos=np.array((.3, .3, .7)))
    pln_DE = PlaneDE(normal=np.array((1., 1., 1.)), pos=-.5*np.ones((3,)))
    comp_DE = ComposeDE([sph_DE, sph2_DE, pln_DE])

    origin = np.array((4.,2.,1.))
    target = np.array((0.,0.,0.))
    light = np.array((1.,1.,1.))
    eye = np.array((-.1,.2,.0))

    shape = (800, 700)
    diagonal = 3.

    stereo_kws = (dict(screen=Screen(origin=origin+eye,
                                     target=target,
                                     shape=shape,
                                     diag=diagonal), 
                       DE=comp_DE, 
                       light=light),
                  dict(screen=Screen(origin=origin-eye,
                                     target=target,
                                     shape=shape,
                                     diag=diagonal), 
                       DE=comp_DE, 
                       light=light))
    
    imageL, imageR = stereo(*stereo_kws)

    # image = (render(**stereo_kws[0]))

    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.6f} seconds.")
    # print(f'image size {image.shape}')
    import matplotlib.pyplot as plt
    plt.imshow(np.concatenate([imageL,imageR], axis=1))
    # plt.imshow(image)
    plt.show()