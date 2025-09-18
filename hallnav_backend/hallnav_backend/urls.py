"""
URL configuration for hallnav_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#from django.contrib import admin
#from django.urls import path, include
#from django.urls import path
#from recognition import views # <-- Change this line


#urlpatterns = [
    
    #path('admin/', admin.site.urls),
    #path('', include('recognition.urls')),  # This line includes your API endpoint] 
   # path('', views.home_page, name='home'), 
    #path('admin/', admin.site.urls),
    #path('api/recognize_hall/', views.recognize_hall, name='recognize_hall'),

#]

from django.urls import path, re_path
from django.contrib import admin
from recognition import views
from django.conf import settings
from django.conf.urls.static import static
from recognition.views import HomePageView, recognize_hall 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/recognize_hall/', views.recognize_hall, name='recognize_hall'),

    # This line is crucial for serving the index.html
    re_path(r'^.*$', HomePageView.as_view(), name='home_page'),

]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)