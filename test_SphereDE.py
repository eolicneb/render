import render_sincrono as rs
import numpy as np

# centre = np.array((2.,-4.,1.))
centre = np.array((-2.,4.,-1.))

sph = rs.SphereDE(radius=25., pos=centre)

# Try the Sphere distance finder with
# a Pytagorean cuaternion to get all
# whole numbers both in position and normal.
# i.e.: 15**2 + 12**2 + 16**2 = 25**2

sur = np.array((0,0,25))
print(f'surface point: {centre+sur}')
dir_ = np.array((12,-16,15))

d = -dir_/np.linalg.norm(dir_)
print('check direction versor:', np.linalg.norm(d))
print('check direction:', (dir_*d).sum()/np.linalg.norm(dir_))

p = centre + sur + 2*dir_

result = sph((p,d))

print("dist: {}, color: {}, normal: {}".format(*result))

distance, color, normal = result
surface = p + distance*d
r_radius = surface - centre
r_radius_norm = np.linalg.norm(r_radius)
print('resulting point: ', surface)
print('resulting radius: ', r_radius_norm, r_radius)

print()
separation = (p-centre)
dist = np.linalg.norm(separation)
print('separation: ', separation, dist)
product = -d*separation
product_sum = product.sum()
cos_ang = product_sum/dist
print(f'd*sep {product}, sum {product_sum}, cos(ang)={cos_ang}, ang={np.arccos(cos_ang)/np.pi}')

def tri_angles(points):
    from itertools import permutations
    angles = []
    print('points:')
    for p in points:
        print(f'\t{p}')
    for a, b, c in permutations(points):
        r, s = b-a, c-a
        cos_ang = (r*s).sum()/np.linalg.norm(r)/np.linalg.norm(s)
        angles.append(np.arccos(cos_ang)/np.pi)
        print('angle: ', angles[-1])
    print('check: ',np.sum(angles))

tri_angles((p, centre, sur+centre))
tri_angles((p, centre, surface))
