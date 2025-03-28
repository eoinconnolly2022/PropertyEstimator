--
-- PostgreSQL database dump
--

-- Dumped from database version 16.8 (Ubuntu 16.8-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.8 (Ubuntu 16.8-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: apikeys; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.apikeys (
    api_key_id integer NOT NULL,
    user_id integer NOT NULL,
    api_key character varying(255) NOT NULL,
    created_at timestamp without time zone NOT NULL,
    expires_at timestamp without time zone
);


--
-- Name: apikeys_api_key_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.apikeys_api_key_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: apikeys_api_key_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.apikeys_api_key_id_seq OWNED BY public.apikeys.api_key_id;


--
-- Name: usagedata; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.usagedata (
    usage_id integer NOT NULL,
    api_key_id integer NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    endpoint character varying(255) NOT NULL,
    request_data text,
    response_data text
);


--
-- Name: usagedata_usage_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.usagedata_usage_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: usagedata_usage_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.usagedata_usage_id_seq OWNED BY public.usagedata.usage_id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    user_id integer NOT NULL,
    username character varying(255) NOT NULL,
    password_hash character varying(255) NOT NULL
);


--
-- Name: users_user_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.users_user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: users_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.users_user_id_seq OWNED BY public.users.user_id;


--
-- Name: apikeys api_key_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.apikeys ALTER COLUMN api_key_id SET DEFAULT nextval('public.apikeys_api_key_id_seq'::regclass);


--
-- Name: usagedata usage_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.usagedata ALTER COLUMN usage_id SET DEFAULT nextval('public.usagedata_usage_id_seq'::regclass);


--
-- Name: users user_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users ALTER COLUMN user_id SET DEFAULT nextval('public.users_user_id_seq'::regclass);


--
-- Name: apikeys apikeys_api_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.apikeys
    ADD CONSTRAINT apikeys_api_key_key UNIQUE (api_key);


--
-- Name: apikeys apikeys_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.apikeys
    ADD CONSTRAINT apikeys_pkey PRIMARY KEY (api_key_id);


--
-- Name: usagedata usagedata_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.usagedata
    ADD CONSTRAINT usagedata_pkey PRIMARY KEY (usage_id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (user_id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: apikeys apikeys_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.apikeys
    ADD CONSTRAINT apikeys_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id);


--
-- Name: usagedata usagedata_api_key_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.usagedata
    ADD CONSTRAINT usagedata_api_key_id_fkey FOREIGN KEY (api_key_id) REFERENCES public.apikeys(api_key_id);


--
-- PostgreSQL database dump complete
--

