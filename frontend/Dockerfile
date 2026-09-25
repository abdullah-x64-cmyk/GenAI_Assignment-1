FROM node:22-alpine AS build

WORKDIR /workspace/app
COPY app/package.json ./package.json
RUN npm install
COPY app/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /workspace/app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
