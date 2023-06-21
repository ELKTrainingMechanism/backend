from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.middleware import csrf
import json
import subprocess

def get_csrf_token(request):
    csrf_token = csrf.get_token(request)
    return JsonResponse({'csrfToken': csrf_token})

@csrf_exempt
def post_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_data = str(data.get('inputValue'))
        output_value = subprocess.check_output(['python', 'scripts/main.py', input_data])
        output_value = output_value.decode('utf-8')
         # Process the input data    
        response_data = {
        'message': 'this is react replying: ' + input_data,
        'inputData': output_value + ', and we are done with the backend, too!',
         }
        return JsonResponse(response_data)

@csrf_exempt
def post_custom_training_args(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_data = str(data.get('customTrainingArgs'))
        input_data = input_data.replace(" ","")
        output_value = subprocess.check_output(['python', 'scripts/replace_args.py', input_data])
        output_value = output_value.decode('utf-8')
        # Process the input data    
        response_data = {
        'message': 'You have submitted the following arguments: ' + str(input_data),
        'output' : output_value,
        }
        return JsonResponse(response_data)


