import jax

if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    print("Hello, world!")
