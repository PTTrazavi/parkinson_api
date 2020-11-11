from django.shortcuts import render, get_object_or_404
from .models import Imageupload
from .serializers import ImageuploadSerializer
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.parsers import FileUploadParser
#from rest_framework.views import APIView
from .util import imgtool, p_detection

# Create your views here.
class ImageuploadViewSet(viewsets.ModelViewSet):
    queryset = Imageupload.objects.all()
    serializer_class = ImageuploadSerializer

    @action(detail=True, methods=['post'])
    def img_process(self, request, pk=None):
        image = get_object_or_404(Imageupload, pk=pk)
        image.readiness = "1"
        image.save()
        serializer = ImageuploadSerializer(image)
        #imgtool(pk)
        p_detection(pk)

        return Response(serializer.data)
"""
class ImageuploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        file_serializer = ImageuploadSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
"""
