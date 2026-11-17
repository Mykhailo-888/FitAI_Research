from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('fitness', '0003_stress_and_input_sources'),
    ]

    operations = [
        migrations.AddField(
            model_name='userdata',
            name='browser_scope',
            field=models.UUIDField(
                blank=True,
                db_index=True,
                null=True,
            ),
        ),
    ]
