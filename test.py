import json
from flask import Flask, Response, request, jsonify
from flask import abort 

app = Flask(__name__)
def first_function():
  try:
    second_function()
  except Exception as error:
    error_message = str(error).split("\n")[0]
    raise ValueError(error_message)

def second_function():
  try:
    print("second_function")
    third_function()
    print("I do not want to see this sentence.")
  except Exception as error:
    error_message = str(error).split("\n")[0]
    raise ValueError(error_message)

def third_function():
  try:
    print("third_function")
    print(1/0)
  except Exception as error:
    error_message = str(error).split("\n")[0]
    raise ValueError(error_message)

@app.route('/control', methods=[ 'POST'])
def get():
  response = ''
  try:
    first_function()
    return jsonify(response)
  except Exception as error:
    error_message = str(error).split("\n")[0]
    response = {"text" : "Something went wrong when operating 'get' process. -> " + str(error_message)}
    print("Error on main function ", error_message)
    return   error_message
  

if __name__ == '__main__':
    app.run(debug=True, host = '127.0.0.1', port = 5000)