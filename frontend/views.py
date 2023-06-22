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
        dict_values = output_value.split(',')

        def get_value(string_arg):
            colon_index = string_arg.index(':')
            substring = string_arg[colon_index + 2:]
            integer_value = float(substring)
            return integer_value

        list_values = []    

        for dict_value in dict_values:
            list_values.append(get_value(dict_value))

        output_value = {
            'small_training_loss': list_values[0],
            'small_validation_loss': list_values[1],
            'small_perplexity': list_values[2],
            'large_training_loss': list_values[3],
            'large_validation_loss': list_values[4],
            'large_perplexity': list_values[5],
            'scaled_training_loss': list_values[6],
            'scaled_validation_loss': list_values[7],
            'scaled_perplexity': list_values[8],
        }    

        response_data = {
        'message': str(input_data),
        'metrics' : output_value,
        }
        
        return JsonResponse(response_data)


