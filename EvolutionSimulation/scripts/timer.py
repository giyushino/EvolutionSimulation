import time 

def timed(function):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = function(*args, **kwargs)  # Call the original function
        t1 = time.perf_counter()
        print(f"{function.__name__} took {(t1 - t0):.6f} seconds to complete")
        #print(f"Took {(t1 - t0):.6f} seconds to complete {function.__name__}\n")
        return result  # Return the result of the original function
    return wrapper
 
