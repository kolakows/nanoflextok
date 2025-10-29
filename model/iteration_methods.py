def rk4_step(f, # function that takes (y,t) and returns dy/dt, i.e. velocity
             y, # current location
             t, # current t value
             dt, # requested time step size 
             ):
    k1 =  f(y, t)
    k2 =  f(y - dt*k1/2, t + dt/2) 
    k3 =  f(y - dt*k2/2, t + dt/2) 
    k4 =  f(y - dt*k3, t + dt) 
    return (k1 + 2*k2 + 2*k3 + k4)/6

def fwd_euler_step(model, current_points, current_t, dt):
    velocity = model(current_points, current_t)
    return velocity
