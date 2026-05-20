from django.urls import path
from . import views

urlpatterns = [

    # HOME
    path('', views.home_view, name='home'),

    # RESEARCH PAGES
    path('stochastic-dynamics/', views.stochastic_dynamics, name='stochastic_dynamics'),

    path('hjb-predictor/', views.hjb_predictor, name='hjb_predictor'),

    path('latent-states/', views.latent_states, name='latent_states'),

    path('custom-neural-network/', views.custom_neural_network, name='custom_neural_network'),

    path('signal-flow/', views.signal_flow, name='signal_flow'),

    path('response-analysis/', views.response_analysis, name='response_analysis'),

    path('dual-layer/', views.dual_layer_evaluation, name='dual_layer_evaluation'),

    path('monitoring-dashboard/', views.monitoring_dashboard, name='monitoring_dashboard'),

    path('opencv-analysis/', views.opencv_analysis, name='opencv_analysis'),

    path('physiological-parameters/', views.physiological_parameters, name='physiological_parameters'),

    # EXISTING PAGES
    path('onboarding/', views.onboarding, name='onboarding'),

    path('history/', views.history, name='history'),

    path('metrics/', views.metrics, name='metrics'),

    path('training-logs/', views.training_log_view, name='training_logs'),

    path('update/', views.update_data, name='update_data'),
    path('research-overview/',views.research_overview,name='research_overview'),

    path('models-overview/', views.models_overview, name='models_overview'),

    path('about-fitai/', views.about_fitai, name='about_fitai'),

    path( 'dashboard-overview/', views.dashboard_overview, name='dashboard_overview'),

    path("custom-nn/", views.custom_nn_view),

    path("hjb/", views.hjb_view),

    path("latent/", views.latent_view),

    path("hypercomplex/", views.hypercomplex_view),

]