from django.core.management.base import BaseCommand

from fitness.dataset_service import inspect_and_register_local_datasets


class Command(BaseCommand):
    help = "Inspect and register every dataset shipped with this repository."

    def handle(self, *args, **options):
        for dataset in inspect_and_register_local_datasets():
            self.stdout.write(
                f"{dataset.name}: {dataset.record_count} rows; "
                f"real={dataset.is_real_data}; synthetic={dataset.is_synthetic}"
            )
