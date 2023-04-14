from fastapi import FastAPI

# initialize the Fast API Application.
app = FastAPI(debug=True)


@app.get("/")
def read_root():
    """Root Message.

    :return: Welcome Message
    :rtype: string
    """
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"
