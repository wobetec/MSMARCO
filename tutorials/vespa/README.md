# Vespa

The objective of this tutorial is to run Vespa directly on Docker without python API interface.

## Vespa Setup

First of all we need to create a docker network to run Vespa.

```bash
docker pull vespaengine/vespa
```

Now let's create a docker network to run Vespa, but loading the scheme we're going to use for our dataset. Make sure to be at `tutorials/vespa` folder to correct load the docker-compose file.

```bash
docker compose up -d
```
Verify if the vespa container is running:

```bash
docker exec vespa vespa status deploy --wait 300
```

Build the vespa application:

```bash
docker exec vespa vespa deploy --wait 300 app-vespa
```

## Feeding Documents

To feed documents to Vespa, we need to run the following command:

```bash
python feed.py copy [none, lower, full]
```

This will copy documents to inside the container and them run `vespa feed` it will be times faster than use the vespa API. But if you want to use the API, you can run the following command:

```bash
python feed.py api
```

## Querying Vespa

To query Vespa, we need to run the following command:

```bash
python query.py Hello World!
```

Just replace `Hello World!` with the query you want to run. The results will be printed in the console.
