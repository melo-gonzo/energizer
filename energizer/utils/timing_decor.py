import time


def track_it(log_file="function_timings.txt"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time

            with open(log_file, "a") as f:
                f.write(
                    f"Function: {func.__name__}, Execution Time: {elapsed_time:.10f} seconds\n"
                )

            return result

        return wrapper

    return decorator
