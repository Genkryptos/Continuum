Tech Stack -> Postgres + pgvector


docker run --name my-pgvector \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=mydb \
  -p 5433:5432 \
  -d ankane/pgvector


