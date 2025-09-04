
create table aka_name (
    id integer not null,
    person_id integer not null,
    name character varying,
    imdb_index varchar(3),
    name_pcode_cf varchar(11),
    name_pcode_nf varchar(11),
    surname_pcode varchar(11),
    md5sum varchar(65),
    primary key (id)
);
create table aka_title (
    id integer not null,
    movie_id integer not null,
    title character varying,
    imdb_index varchar(4),
    kind_id integer not null,
    production_year integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note varchar(72),
    md5sum varchar(32),
    primary key (id)
);
create table cast_info (
    id integer not null,
    person_id integer not null,
    movie_id integer not null,
    person_role_id integer,
    note character varying,
    nr_order integer,
    role_id integer not null,
    primary key (id)
);
create table char_name (
    id integer not null,
    name character varying,
    imdb_index varchar(2),
    imdb_id integer,
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(32),
    primary key (id)
);
create table comp_cast_type (
    id integer not null,
    kind varchar(32) not null,
    primary key (id)
);
create table company_name (
    id integer not null,
    name character varying,
    country_code varchar(6),
    imdb_id integer,
    name_pcode_nf varchar(5),
    name_pcode_sf varchar(5),
    md5sum varchar(32),
    primary key (id)
);
create table company_type (
    id integer not null,
    kind varchar(32),
    primary key (id)
);
create table complete_cast (
    id integer not null,
    movie_id integer,
    subject_id integer not null,
    status_id integer not null,
    primary key (id)
);
create table info_type (
    id integer not null,
    info varchar(32) not null,
    primary key (id)
);
create table keyword (
    id integer not null,
    keyword character varying not null,
    phonetic_code varchar(5),
    primary key (id)
);
create table kind_type (
    id integer not null,
    kind varchar(15),
    primary key (id)
);
create table link_type (
    id integer not null,
    link varchar(32) not null,
    primary key (id)
);
create table movie_companies (
    id integer not null,
    movie_id integer not null,
    company_id integer not null,
    company_type_id integer not null,
    note character varying,
    primary key (id)
);
create table movie_info_idx (
    id integer not null,
    movie_id integer not null,
    info_type_id integer not null,
    info character varying not null,
    note varchar(1),
    primary key (id)
);
create table movie_keyword (
    id integer not null,
    movie_id integer not null,
    keyword_id integer not null,
    primary key (id)
);
create table movie_link (
    id integer not null,
    movie_id integer not null,
    linked_movie_id integer not null,
    link_type_id integer not null,
    primary key (id)
);
create table name (
    id integer not null,
    name character varying not null,
    imdb_index varchar(9),
    imdb_id integer,
    gender varchar(1),
    name_pcode_cf varchar(5),
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(32),
    primary key (id)
);
create table role_type (
    id integer not null,
    role varchar(32) not null,
    primary key (id)
);
create table title (
    id integer not null,
    title character varying,
    imdb_index varchar(5),
    kind_id integer not null,
    production_year integer,
    imdb_id integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years varchar(49),
    md5sum varchar(32),
    primary key (id)
);
create table movie_info (
    id integer not null,
    movie_id integer not null,
    info_type_id integer not null,
    info character varying,
    note character varying,
    primary key (id)
);
create table person_info (
    id integer not null,
    person_id integer not null,
    info_type_id integer not null,
    info character varying,
    note character varying,
    primary key (id)
);