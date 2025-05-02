import pytest
from google.auth import default
from google.cloud import bigquery


@pytest.mark.skip("BigQuery tests require credentials")
def test_bq():
    """
    Tests behavior of BigQuery.

    To run this test:

    - ``gcloud auth application-default login``
    - place credentials in ``~/.config/gcloud/application_default_credentials.json``

    """
    credentials, project = default()
    client = bigquery.Client(credentials=credentials, project=project)

    # Construct a query
    query = """
        SELECT name, SUM(number) as total_people
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE state = 'TX'
        GROUP BY name
        ORDER BY total_people DESC
        LIMIT 20
    """

    # Run the query
    query_job = client.query(query)

    # Get results
    results = query_job.result()

    # Print results
    for row in results:
        print(f"{row.name}: {row.total_people}")


@pytest.mark.skip("BigQuery tests require credentials")
def test_sra():
    """
    Tests behavior of BigQuery.

    To run this test:

    - ``gcloud auth application-default login``
    - place credentials in ``~/.config/gcloud/application_default_credentials.json``

    """
    credentials, project = default()
    client = bigquery.Client(credentials=credentials, project=project)

    # Construct a query
    query = """
        SELECT *
        FROM `nih-sra-datastore.sra.metadata` as s
        WHERE organism = 'Syngnathus scovelli' and ( ('sex_calc', 'female') in 
         UNNEST(s.attributes) and ('dev_stage_sam', 'Adult') in UNNEST(s.attributes) ) limit 10
    """

    # Run the query
    query_job = client.query(query)

    # Get results
    results = query_job.result()

    # Print results
    for row in results:
        print(type(row))
        print(f"{row.organism}: {row.attributes}")
        for a in row.attributes:
            print(f"  {a['k']} == {a['v']}")
