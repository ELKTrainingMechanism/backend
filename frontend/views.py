from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
import json
import subprocess


class MyEndpoint(View):

    def post(self, request):
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
def get_csrf_token(request):
    if request.method == 'GET':
        csrf_token = get_token(request)
        return JsonResponse({'csrfToken': csrf_token})

def index(request):
    return render(request, 'index.html')
